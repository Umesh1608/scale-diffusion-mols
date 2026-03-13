"""Tests for the Lanczos algorithm and posterior sampling (molssd.core.lanczos)."""
import torch
import pytest

from molssd.core.lanczos import (
    lanczos_tridiagonalization,
    posterior_covariance_eigendecomp,
    sample_non_isotropic,
)


# -------------------------------------------------------------------
# Lanczos tridiagonalization
# -------------------------------------------------------------------


class TestLanczosTridiagonalization:

    def test_lanczos_identity(self, seed):
        """Lanczos on the identity matrix recovers eigenvalue 1."""
        n = 10
        k = 5

        def matvec(v):
            return v  # Identity

        T_k, Q_k = lanczos_tridiagonalization(matvec, n, k)

        # Eigenvalues of T_k should all be 1
        eigenvalues = torch.linalg.eigvalsh(T_k)
        expected = torch.ones(k)

        assert torch.allclose(eigenvalues, expected, atol=1e-5), (
            f"Identity matrix eigenvalues should be 1, got {eigenvalues}"
        )

    def test_lanczos_diagonal(self, seed):
        """Lanczos on a diagonal matrix recovers eigenvalues."""
        n = 8
        k = 8  # Full decomposition
        diag_vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        A = torch.diag(diag_vals)

        def matvec(v):
            return A @ v

        T_k, Q_k = lanczos_tridiagonalization(matvec, n, k)

        # Eigenvalues of T_k should match diag_vals (when sorted)
        eigenvalues = torch.linalg.eigvalsh(T_k)
        expected = diag_vals.sort().values

        assert torch.allclose(eigenvalues, expected, atol=1e-3), (
            f"Diagonal matrix eigenvalues mismatch: {eigenvalues} vs {expected}"
        )

    def test_lanczos_orthogonality(self, seed):
        """Q_k columns are orthonormal."""
        n = 15
        k = 8

        # Use a random symmetric matrix
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        def matvec(v):
            return A @ v

        T_k, Q_k = lanczos_tridiagonalization(matvec, n, k)

        # Q_k^T Q_k should be identity
        QtQ = Q_k.T @ Q_k
        I_k = torch.eye(Q_k.shape[1])

        assert torch.allclose(QtQ, I_k, atol=1e-5), (
            f"Lanczos vectors not orthonormal: max deviation = "
            f"{(QtQ - I_k).abs().max().item()}"
        )

    def test_lanczos_tridiagonal(self, seed):
        """T_k is symmetric tridiagonal."""
        n = 12
        k = 6

        A = torch.randn(n, n)
        A = (A + A.T) / 2

        def matvec(v):
            return A @ v

        T_k, Q_k = lanczos_tridiagonalization(matvec, n, k)
        k_eff = T_k.shape[0]

        # Symmetric
        assert torch.allclose(T_k, T_k.T, atol=1e-6), "T_k must be symmetric"

        # Tridiagonal: entries more than 1 away from diagonal should be zero
        for i in range(k_eff):
            for j in range(k_eff):
                if abs(i - j) > 1:
                    assert abs(T_k[i, j].item()) < 1e-6, (
                        f"T_k[{i},{j}] = {T_k[i,j].item()} should be zero "
                        f"(tridiagonal)"
                    )


# -------------------------------------------------------------------
# Posterior covariance eigendecomposition
# -------------------------------------------------------------------


class TestPosteriorEigendecomp:

    def test_posterior_eigenvalues_positive(self, seed):
        """All eigenvalues of the posterior covariance Sigma are positive."""
        n = 10
        k = 4

        # Build a low-rank symmetric PSD matrix for M_t^T M_t
        B = torch.randn(k, n)
        MtTMt = B.T @ B  # PSD, rank <= k

        def matvec(v):
            return MtTMt @ v

        sigma_t = torch.tensor(0.8)
        sigma_tm1 = torch.tensor(0.6)

        eigenvalues, ritz_vectors = posterior_covariance_eigendecomp(
            MtT_Mt_matvec=matvec,
            sigma_t=sigma_t,
            sigma_t_minus_1=sigma_tm1,
            n=n,
            k=k,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert (eigenvalues > 0).all(), (
            f"Posterior eigenvalues must be positive, got {eigenvalues}"
        )


# -------------------------------------------------------------------
# Non-isotropic sampling
# -------------------------------------------------------------------


class TestSampleNonIsotropic:

    def test_sample_non_isotropic_shape(self, seed):
        """Output shape matches mu shape."""
        N, D = 10, 3
        k = 4

        mu = torch.randn(N, D)
        eigenvalues = torch.rand(k) + 0.1  # positive
        ritz_vectors = torch.linalg.qr(torch.randn(N, k))[0]  # orthonormal
        sigma_tm1 = torch.tensor(0.5)

        sample = sample_non_isotropic(mu, eigenvalues, ritz_vectors, sigma_tm1)

        assert sample.shape == mu.shape, (
            f"Expected shape {mu.shape}, got {sample.shape}"
        )

    def test_sample_non_isotropic_mean(self, seed):
        """Empirical mean over many samples is approximately mu."""
        N, D = 8, 3
        k = 3
        num_samples = 5000

        mu = torch.randn(N, D)
        # Use eigenvalues close to sigma_tm1^2 so the correction is small
        sigma_tm1 = torch.tensor(0.5)
        eigenvalues = torch.full((k,), sigma_tm1.item() ** 2)
        ritz_vectors = torch.linalg.qr(torch.randn(N, k))[0]

        samples = torch.stack([
            sample_non_isotropic(mu, eigenvalues, ritz_vectors, sigma_tm1)
            for _ in range(num_samples)
        ])  # (num_samples, N, D)

        empirical_mean = samples.mean(dim=0)

        # With eigenvalues == sigma_tm1^2, the correction_coeffs are zero,
        # so the mean should be very close to mu
        assert torch.allclose(empirical_mean, mu, atol=0.1), (
            f"Empirical mean deviates from mu: "
            f"max diff = {(empirical_mean - mu).abs().max().item()}"
        )
