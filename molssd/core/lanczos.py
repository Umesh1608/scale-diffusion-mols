"""Lanczos algorithm for efficient sampling from non-isotropic posteriors.

At resolution-changing steps in the MolSSD reverse process, the posterior
covariance is *not* a scalar multiple of the identity.  Instead it has the
form:

    Sigma_{t -> t-1} = sigma_{t-1}^2 I  -  (sigma_{t-1}^4 / sigma_t^2) M_t^T M_t

where M_t is the per-step degradation operator.  Because M_t^T M_t is low-rank
(rank = N_{r(t)}, the number of nodes at the coarser resolution), only a small
number of directions deviate from isotropic.  The Lanczos algorithm lets us
compute an approximate eigendecomposition of M_t^T M_t using only matrix-vector
products, without ever forming the full matrix.

Sampling strategy:
    1. Run Lanczos on M_t^T M_t to get k Ritz pairs (lambda_j, v_j).
    2. Transform to eigenvalues of Sigma:  d_j = sigma_{t-1}^2 - (sigma_{t-1}^4 / sigma_t^2) * lambda_j.
    3. Sample:  x = mu + sigma_{t-1} z + V_k diag(sqrt(d_j) - sigma_{t-1}) V_k^T z
       where z ~ N(0, I).  This corrects only the k directions that differ
       from the isotropic baseline.

References:
    - Golub & Van Loan, "Matrix Computations", Ch. 10 (Lanczos algorithm).
    - SSD paper (arXiv:2603.08709), Appendix B (posterior sampling).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Lanczos tridiagonalization
# ---------------------------------------------------------------------------


def lanczos_tridiagonalization(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    k: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lanczos tridiagonalization of a symmetric matrix via matrix-vector products.

    Given a symmetric matrix A accessible only through its action ``v -> A v``,
    produces the Lanczos decomposition:

        A Q_k ~ Q_k T_k

    where T_k is a k x k symmetric tridiagonal matrix and Q_k is an n x k
    matrix whose columns are orthonormal Lanczos vectors.

    Full reorthogonalization (modified Gram-Schmidt against all previous
    vectors) is applied at every step for numerical stability.  This costs
    O(n k^2) but k is typically very small (equal to the number of super-nodes
    at the coarser resolution, e.g. 3-10 for QM9 molecules).

    Args:
        matvec_fn: A callable that computes ``A @ v`` for a vector ``v`` of
            shape ``(n,)``.  The underlying matrix A must be symmetric.
        n: Dimension of the matrix (length of vectors).
        k: Number of Lanczos iterations.  Clamped to ``[0, n]``.
        device: Torch device for the output tensors.  Defaults to CPU.
        dtype: Torch dtype for the output tensors.  Defaults to float32.

    Returns:
        T_k: Symmetric tridiagonal matrix of shape ``(k, k)``.
        Q_k: Matrix of Lanczos vectors of shape ``(n, k)``, columns orthonormal.

    Notes:
        When ``k == 0``, returns empty tensors of shape ``(0, 0)`` and ``(n, 0)``.
        When ``k >= n``, ``k`` is clamped to ``n`` (full decomposition).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    # Clamp k to valid range
    k = max(0, min(k, n))

    if k == 0:
        T = torch.zeros(0, 0, device=device, dtype=dtype)
        Q = torch.zeros(n, 0, device=device, dtype=dtype)
        return T, Q

    # Allocate storage
    alphas = torch.zeros(k, device=device, dtype=dtype)  # diagonal of T
    betas = torch.zeros(k, device=device, dtype=dtype)    # sub/super-diagonal
    Q = torch.zeros(n, k, device=device, dtype=dtype)

    # Initial random vector, normalized
    q = torch.randn(n, device=device, dtype=dtype)
    q = q / q.norm().clamp(min=1e-12)
    Q[:, 0] = q

    # r for the iteration (will be updated)
    r = matvec_fn(q)
    alpha = torch.dot(q, r)
    alphas[0] = alpha
    r = r - alpha * q

    for j in range(1, k):
        # Full reorthogonalization: modified Gram-Schmidt against all
        # previous Lanczos vectors to combat loss of orthogonality.
        for i in range(j):
            r = r - torch.dot(Q[:, i], r) * Q[:, i]

        beta = r.norm()
        betas[j] = beta

        if beta < 1e-12:
            # Invariant subspace found; generate a new random vector
            # orthogonal to existing Q columns.
            r = torch.randn(n, device=device, dtype=dtype)
            for i in range(j):
                r = r - torch.dot(Q[:, i], r) * Q[:, i]
            beta_new = r.norm()
            if beta_new < 1e-12:
                # Entire space has been spanned; truncate
                T = torch.zeros(j, j, device=device, dtype=dtype)
                for idx in range(j):
                    T[idx, idx] = alphas[idx]
                    if idx > 0:
                        T[idx, idx - 1] = betas[idx]
                        T[idx - 1, idx] = betas[idx]
                return T, Q[:, :j]
            r = r / beta_new

        else:
            r = r / beta

        q = r
        Q[:, j] = q

        # Matrix-vector product
        r = matvec_fn(q)
        alpha = torch.dot(q, r)
        alphas[j] = alpha

        # Three-term recurrence
        r = r - alpha * q - beta * Q[:, j - 1]

    # Build the tridiagonal matrix T_k
    T = torch.zeros(k, k, device=device, dtype=dtype)
    for j in range(k):
        T[j, j] = alphas[j]
        if j > 0:
            T[j, j - 1] = betas[j]
            T[j - 1, j] = betas[j]

    return T, Q


# ---------------------------------------------------------------------------
# Posterior covariance eigendecomposition
# ---------------------------------------------------------------------------


def posterior_covariance_eigendecomp(
    MtT_Mt_matvec: Callable[[torch.Tensor], torch.Tensor],
    sigma_t: torch.Tensor,
    sigma_t_minus_1: torch.Tensor,
    n: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the approximate eigendecomposition of the posterior covariance.

    The posterior covariance at a resolution-changing step is:

        Sigma = sigma_{t-1}^2 I  -  (sigma_{t-1}^4 / sigma_t^2) M_t^T M_t

    This function uses Lanczos to find the dominant eigenpairs of M_t^T M_t,
    then transforms them to eigenpairs of Sigma.

    The Ritz vectors of M_t^T M_t are also eigenvectors of Sigma (since Sigma
    is a linear combination of I and M_t^T M_t).  Only the eigenvalues change:

        d_j = sigma_{t-1}^2  -  (sigma_{t-1}^4 / sigma_t^2) * lambda_j

    where lambda_j are the Ritz values of M_t^T M_t.  Eigenvalues are clamped
    to a minimum of epsilon = 1e-6 to ensure the covariance is positive
    semi-definite (accounting for numerical error).

    Args:
        MtT_Mt_matvec: Callable computing ``(M_t^T M_t) @ v`` for a vector
            ``v`` of shape ``(n,)``.  See
            :meth:`~molssd.core.degradation.DegradationOperator.compute_MtT_Mt_matvec`.
        sigma_t: Noise standard deviation at timestep t (scalar tensor).
        sigma_t_minus_1: Noise standard deviation at timestep t-1 (scalar tensor).
        n: Dimension of the space (number of nodes at resolution r(t-1)).
        k: Number of Lanczos iterations (typically = N_{r(t)}, the number of
            nodes at the coarser level).
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        eigenvalues: Tensor of shape ``(k_eff,)`` -- the eigenvalues d_j of
            Sigma in the Lanczos subspace.  ``k_eff <= k`` (may be smaller if
            an invariant subspace was found early).
        ritz_vectors: Tensor of shape ``(n, k_eff)`` -- the corresponding
            approximate eigenvectors of Sigma (same as Ritz vectors of
            M_t^T M_t).
    """
    epsilon = 1e-6

    sigma_t_val = sigma_t.detach().float()
    sigma_tm1_val = sigma_t_minus_1.detach().float()

    sigma_t_sq = sigma_t_val ** 2
    sigma_tm1_sq = sigma_tm1_val ** 2
    sigma_tm1_4 = sigma_tm1_val ** 4

    # Scaling factor for M_t^T M_t contribution
    scale = sigma_tm1_4 / sigma_t_sq.clamp(min=1e-12)

    # Run Lanczos on M_t^T M_t
    T_k, Q_k = lanczos_tridiagonalization(
        matvec_fn=MtT_Mt_matvec,
        n=n,
        k=k,
        device=device,
        dtype=dtype,
    )

    k_eff = T_k.shape[0]
    if k_eff == 0:
        return (
            torch.zeros(0, device=device, dtype=dtype),
            torch.zeros(n, 0, device=device, dtype=dtype),
        )

    # Eigendecompose the small tridiagonal matrix T_k
    # T_k = S diag(lambda) S^T
    lambdas, S = torch.linalg.eigh(T_k)  # lambdas shape (k_eff,), S shape (k_eff, k_eff)

    # Transform to eigenvectors of the full space: V = Q_k @ S
    ritz_vectors = Q_k @ S  # (n, k_eff)

    # Transform eigenvalues: d_j = sigma_{t-1}^2 - scale * lambda_j
    eigenvalues = sigma_tm1_sq - scale * lambdas

    # Clamp to ensure positive semi-definiteness
    eigenvalues = eigenvalues.clamp(min=epsilon)

    return eigenvalues, ritz_vectors


# ---------------------------------------------------------------------------
# Non-isotropic sampling
# ---------------------------------------------------------------------------


def sample_non_isotropic(
    mu: torch.Tensor,
    eigenvalues: torch.Tensor,
    ritz_vectors: torch.Tensor,
    sigma_t_minus_1: torch.Tensor,
) -> torch.Tensor:
    """Sample from a non-isotropic Gaussian posterior at a resolution-changing step.

    The posterior is:

        x_{t-1} ~ N(mu, Sigma)

    where Sigma = sigma_{t-1}^2 I + V_k diag(d_j - sigma_{t-1}^2) V_k^T
    (with V_k the Ritz vectors and d_j the transformed eigenvalues).

    Efficient sampling exploits the fact that only k directions deviate from
    isotropic.  We sample an isotropic Gaussian, then apply a rank-k correction:

        x = mu + sigma_{t-1} z + V_k diag(sqrt(d_j) - sigma_{t-1}) V_k^T z

    where z ~ N(0, I).  The first term provides the isotropic baseline,
    and the second corrects the k special directions to have the right
    variance sqrt(d_j) instead of sigma_{t-1}.

    Each spatial dimension (column of mu) is sampled independently with the
    same covariance structure.

    Args:
        mu: Posterior mean of shape ``(N, D)`` where N is the number of nodes
            at resolution r(t-1) and D is the spatial dimension (typically 3).
        eigenvalues: Tensor of shape ``(k,)`` -- eigenvalues d_j of the
            posterior covariance Sigma.
        ritz_vectors: Tensor of shape ``(N, k)`` -- eigenvectors of Sigma.
        sigma_t_minus_1: Noise standard deviation at timestep t-1 (scalar
            tensor).

    Returns:
        Sampled x_{t-1} of shape ``(N, D)``.
    """
    N, D = mu.shape
    device = mu.device
    dtype = mu.dtype

    sigma_tm1 = sigma_t_minus_1.to(device=device, dtype=dtype)

    # Sample isotropic noise
    z = torch.randn(N, D, device=device, dtype=dtype)

    # Isotropic component: sigma_{t-1} * z
    x = mu + sigma_tm1 * z

    k = eigenvalues.shape[0]
    if k == 0:
        return x

    # Correction coefficients: sqrt(d_j) - sigma_{t-1} for each Ritz direction
    # These adjust the variance from sigma_{t-1}^2 to d_j in direction v_j
    sqrt_d = torch.sqrt(eigenvalues.to(device=device, dtype=dtype))  # (k,)
    correction_coeffs = sqrt_d - sigma_tm1  # (k,)

    # Project z onto the Ritz subspace: V_k^T z  -> shape (k, D)
    # ritz_vectors: (N, k), z: (N, D)
    VkT_z = ritz_vectors.t() @ z  # (k, D)

    # Scale by correction coefficients: diag(correction_coeffs) @ V_k^T z
    # correction_coeffs: (k,), VkT_z: (k, D)
    scaled = correction_coeffs.unsqueeze(1) * VkT_z  # (k, D)

    # Project back to full space: V_k @ scaled  -> (N, D)
    correction = ritz_vectors @ scaled  # (N, D)

    x = x + correction

    return x
