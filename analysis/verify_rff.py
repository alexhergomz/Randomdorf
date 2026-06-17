#!/usr/bin/env python3
"""
Numerical verification of the RFF (Random Fourier Features) ocean model
against the folded-rFFT Tessendorf reference, and against analytic theory.

What this proves:
  (1) Energy law:  Var(h_RFF) == sigma^2 = integral F(k) dk   (no fudge factor)
  (2) Wiener-Khinchin: empirical autocovariance C(r) == integral F(k) J0(kr) dk
  (3) Gaussianity of the synthesized field
  (4) Radial spectrum of the RFF field matches the target F(k)
  (5) Convergence of Schemes A (pure RFF), B (stratified), C (equal-energy) vs M
  (6) RFF vs folded-rFFT look statistically identical

Everything quantitative uses the ISOTROPIC spectrum so the analytic C(r) and F(k)
hold exactly; a DIRECTIONAL run is used only for the pretty heightmaps.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.stats import kurtosis, skew

rng_global = np.random.default_rng(12345)
G = 9.81

# ----------------------------------------------------------------------------
# 1. Spectrum:  Pierson-Moskowitz in frequency -> wavenumber spectrum F(k)
# ----------------------------------------------------------------------------
def pm_omega(omega, U):
    """PM omnidirectional frequency spectrum S(omega) [m^2 s]."""
    alpha = 8.1e-3
    wp = 0.855 * G / U
    out = np.zeros_like(omega)
    m = omega > 1e-6
    out[m] = alpha * G**2 / omega[m]**5 * np.exp(-1.25 * (wp / omega[m])**4)
    return out

def F_k(k, U):
    """Omnidirectional WAVENUMBER spectrum F(k) [m^3], deep water w=sqrt(gk).
       F(k) = S(w(k)) dw/dk,  dw/dk = 0.5 sqrt(g/k)."""
    k = np.asarray(k, float)
    out = np.zeros_like(k)
    m = k > 1e-9
    w = np.sqrt(G * k[m])
    out[m] = pm_omega(w, U) * 0.5 * np.sqrt(G / k[m])
    return out

def total_variance(U, kmax=20.0, n=200000):
    k = np.linspace(1e-6, kmax, n)
    return np.trapezoid(F_k(k, U), k)

# ----------------------------------------------------------------------------
# 2. Wave sampling: Schemes A / B / C
# ----------------------------------------------------------------------------
def build_cdf(U, kmax=20.0, n=20000):
    k = np.linspace(1e-6, kmax, n)
    f = F_k(k, U)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(k))])
    sigma2 = cdf[-1]
    return k, cdf, sigma2

def invert_cdf(k_tab, cdf, targets):
    return np.interp(targets, cdf, k_tab)

def sample_waves(M, U, scheme, directional=False, theta0=0.0, s=4.0, rng=None):
    """Return dict with k (M,2), omega (M,), phase (M,), amp (M,)."""
    if rng is None:
        rng = rng_global
    k_tab, cdf, sigma2 = build_cdf(U)

    if scheme == "A":                      # pure RFF: iid from p(k), equal amp
        u = rng.uniform(0, sigma2, M)
        kmag = invert_cdf(k_tab, cdf, u)
        amp = np.full(M, np.sqrt(2.0 * sigma2 / M))
    elif scheme == "B":                    # stratified RFF, equal amp
        edges = np.linspace(0, sigma2, M + 1)
        u = edges[:-1] + rng.uniform(0, 1, M) * np.diff(edges)
        kmag = invert_cdf(k_tab, cdf, u)
        amp = np.full(M, np.sqrt(2.0 * sigma2 / M))
    elif scheme == "C":                    # deterministic equal-dk grid, amp from spectrum
        kmin, kmax = 1e-3, 2.0                           # classic ocean sum-of-sines
        kedge = np.linspace(kmin, kmax, M + 1)
        kmag = 0.5 * (kedge[:-1] + kedge[1:])           # bin centers
        dk = np.diff(kedge)
        amp = np.sqrt(2.0 * F_k(kmag, U) * dk)          # a_j^2/2 = F(k) dk  (random phase only)
    else:
        raise ValueError(scheme)

    # direction
    if directional:
        th = sample_direction(M, theta0, s, rng)
    else:
        th = rng.uniform(0, 2 * np.pi, M)               # isotropic
    kx = kmag * np.cos(th)
    kz = kmag * np.sin(th)
    omega = np.sqrt(G * kmag)
    phase = rng.uniform(0, 2 * np.pi, M)
    return dict(k=np.stack([kx, kz], 1), kmag=kmag, omega=omega,
                phase=phase, amp=amp, sigma2=sigma2)

def sample_direction(M, theta0, s, rng):
    """cos^{2s} spreading lobe around theta0, via numeric inverse-CDF."""
    th = np.linspace(-np.pi, np.pi, 4000)
    d = np.cos(th / 2.0)**(2 * s)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(th))])
    cdf /= cdf[-1]
    u = rng.uniform(0, 1, M)
    return theta0 + np.interp(u, cdf, th)

# ----------------------------------------------------------------------------
# 3. RFF field synthesis on a grid (the actual model; NO FFT used)
# ----------------------------------------------------------------------------
def rff_field(waves, N, L, t=0.0):
    x = (np.arange(N) * (L / N))
    X, Z = np.meshgrid(x, x, indexing="xy")
    h = np.zeros((N, N))
    k = waves["k"]; amp = waves["amp"]; om = waves["omega"]; ph = waves["phase"]
    for j in range(len(amp)):
        theta = k[j, 0] * X + k[j, 1] * Z - om[j] * t + ph[j]
        h += amp[j] * np.cos(theta)
    return h, X, Z

# ----------------------------------------------------------------------------
# 4. Folded real-valued rFFT reference (the §2 "real Tessendorf")
# ----------------------------------------------------------------------------
def fft_reference(N, L, U, rng=None):
    if rng is None:
        rng = rng_global
    dx = L / N
    kf = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KZ = np.meshgrid(kf, kf, indexing="xy")
    K = np.hypot(KX, KZ)
    dk = 2 * np.pi / L
    Psi = np.zeros_like(K)                       # 2D PSD, isotropic: F(k)/(2 pi k)
    m = K > 1e-9
    Psi[m] = F_k(K[m], U) / (2 * np.pi * K[m])
    A = np.sqrt(Psi * dk * dk) * (rng.standard_normal(K.shape)
                                  + 1j * rng.standard_normal(K.shape)) / np.sqrt(2)
    # h = sqrt(2) Re( sum_k A e^{ik.x} ) ; sum = N^2 * ifft2(A)
    h = np.sqrt(2) * np.real(np.fft.ifft2(A)) * N * N
    return h

# ----------------------------------------------------------------------------
# 5. Measurement: radial periodogram and radial autocovariance (FFT used only
#    to MEASURE the already-synthesized field, not to build it)
# ----------------------------------------------------------------------------
def radial_spectrum(h, L):
    N = h.shape[0]
    dx = L / N
    Hd = np.fft.fft2(h)
    Psi_hat = (dx**2 / ((2 * np.pi)**2 * N**2)) * np.abs(Hd)**2  # integrates to Var
    kf = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KZ = np.meshgrid(kf, kf, indexing="xy")
    K = np.hypot(KX, KZ).ravel()
    P = Psi_hat.ravel()
    dk = 2 * np.pi / L
    nb = N // 2
    kbin = np.arange(nb) * dk
    idx = np.clip((K / dk).astype(int), 0, nb - 1)
    Pmean = np.zeros(nb); cnt = np.zeros(nb)
    np.add.at(Pmean, idx, P); np.add.at(cnt, idx, 1)
    Pmean /= np.maximum(cnt, 1)
    F_hat = 2 * np.pi * kbin * Pmean             # back to omnidirectional F(k)
    return kbin, F_hat

def radial_autocov(h, L):
    N = h.shape[0]
    dx = L / N
    P = np.abs(np.fft.fft2(h))**2
    C2 = np.real(np.fft.ifft2(P)) / (N * N)      # biased autocorr, C2[0,0]=Var
    C2 = np.fft.fftshift(C2)
    c = np.arange(N) - N // 2
    RX, RZ = np.meshgrid(c * dx, c * dx, indexing="xy")
    R = np.hypot(RX, RZ).ravel()
    Cv = C2.ravel()
    nb = N // 2
    rbin = np.arange(nb) * dx
    idx = np.clip((R / dx).astype(int), 0, nb - 1)
    Cmean = np.zeros(nb); cnt = np.zeros(nb)
    np.add.at(Cmean, idx, Cv); np.add.at(cnt, idx, 1)
    Cmean /= np.maximum(cnt, 1)
    return rbin, Cmean

def analytic_autocov(r, U, kmax=20.0, n=40000):
    k = np.linspace(1e-6, kmax, n)
    f = F_k(k, U)
    return np.array([np.trapezoid(f * j0(k * rr), k) for rr in r])

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
def main():
    U = 10.0
    N = 512
    L = 512.0
    M = 256
    sigma2 = total_variance(U)
    Hs = 4 * np.sqrt(sigma2)
    print("=" * 70)
    print(f"PM spectrum, wind U = {U} m/s")
    print(f"  sigma^2 (target variance) = {sigma2:.6f} m^2")
    print(f"  Hs = 4*sqrt(sigma^2)      = {Hs:.4f} m   (PM rule 0.21 U^2/g = {0.21*U**2/G:.4f} m)")
    print("=" * 70)

    results = {}
    for scheme in ["A", "B", "C"]:
        w = sample_waves(M, U, scheme, rng=np.random.default_rng(7))
        h, X, Z = rff_field(w, N, L)
        var = h.var()
        print(f"\nScheme {scheme}:  M={M}")
        print(f"  Var(h_RFF)            = {var:.6f} m^2   "
              f"(ratio to sigma^2 = {var/sigma2:.4f})")
        print(f"  Hs_emp = 4 sqrt(var)  = {4*np.sqrt(var):.4f} m")
        print(f"  mean = {h.mean():+.4e},  skew = {skew(h.ravel()):+.4f},  "
              f"excess kurtosis = {kurtosis(h.ravel()):+.4f}")
        results[scheme] = (w, h)

    # ---- folded rFFT reference ----
    h_ref = fft_reference(N, L, U, rng=np.random.default_rng(99))
    print(f"\nFolded rFFT reference:")
    print(f"  Var(h_ref)            = {h_ref.var():.6f} m^2   "
          f"(ratio to sigma^2 = {h_ref.var()/sigma2:.4f})")

    # ========================= FIGURES =========================
    # Fig 1: heightmaps (directional RFF vs FFT reference)
    wd = sample_waves(M, U, "C", directional=True, theta0=np.deg2rad(35),
                      rng=np.random.default_rng(3))
    hd, _, _ = rff_field(wd, N, L)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    vmax = 2.5 * np.sqrt(sigma2)
    for a, img, ttl in [(ax[0], results["C"][1], "RFF Scheme C (isotropic)"),
                        (ax[1], hd, "RFF Scheme C (directional 35deg)"),
                        (ax[2], h_ref, "Folded rFFT reference")]:
        im = a.imshow(img, cmap="ocean", vmin=-vmax, vmax=vmax,
                      extent=[0, L, 0, L])
        a.set_title(ttl); a.set_xlabel("x [m]"); a.set_ylabel("z [m]")
        plt.colorbar(im, ax=a, fraction=0.046, label="height [m]")
    fig.tight_layout(); fig.savefig("fig1_heightmaps.png", dpi=110)
    print("\n[saved] fig1_heightmaps.png")

    # Fig 2: radial spectrum overlay
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    kk = np.linspace(1e-3, 1.0, 1000)
    for scheme, col in [("B", "tab:blue"), ("C", "tab:green")]:
        kb, Fh = radial_spectrum(results[scheme][1], L)
        ax[0].plot(kb, Fh, color=col, lw=1.2, label=f"RFF {scheme} (measured)")
    ax[0].plot(kk, F_k(kk, U), "k--", lw=2, label="target F(k)")
    ax[0].set_xlim(0, 0.5); ax[0].set_xlabel("k [rad/m]")
    ax[0].set_ylabel("F(k) [m^3]"); ax[0].legend(); ax[0].set_title("Radial spectrum")
    kb, Fh = radial_spectrum(h_ref, L)
    ax[1].plot(kb, Fh, "tab:red", lw=1.2, label="rFFT ref (measured)")
    ax[1].plot(kk, F_k(kk, U), "k--", lw=2, label="target F(k)")
    ax[1].set_xlim(0, 0.5); ax[1].set_xlabel("k [rad/m]")
    ax[1].set_ylabel("F(k) [m^3]"); ax[1].legend()
    ax[1].set_title("rFFT reference spectrum")
    fig.tight_layout(); fig.savefig("fig2_spectrum.png", dpi=110)
    print("[saved] fig2_spectrum.png")

    # Fig 3: autocovariance empirical vs analytic
    fig, ax = plt.subplots(figsize=(8, 5))
    rb, Cb = radial_autocov(results["C"][1], L)
    Can = analytic_autocov(rb[:120], U)
    ax.plot(rb[:120], Cb[:120], "tab:green", lw=1.5, label="RFF C (empirical)")
    ax.plot(rb[:120], Can, "k--", lw=2, label="analytic ∫F(k)J0(kr)dk")
    ax.axhline(sigma2, color="gray", ls=":", label=f"sigma^2 = {sigma2:.3f}")
    ax.set_xlabel("r [m]"); ax.set_ylabel("C(r) [m^2]")
    ax.set_title("Autocovariance: Wiener-Khinchin check"); ax.legend()
    fig.tight_layout(); fig.savefig("fig3_autocov.png", dpi=110)
    print("[saved] fig3_autocov.png")

    # Fig 4: Gaussianity
    fig, ax = plt.subplots(figsize=(8, 5))
    hC = results["C"][1].ravel()
    ax.hist(hC, bins=120, density=True, alpha=0.6, color="tab:green",
            label="RFF height pdf")
    xx = np.linspace(hC.min(), hC.max(), 400)
    ax.plot(xx, np.exp(-xx**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2),
            "k--", lw=2, label=f"N(0, sigma^2={sigma2:.3f})")
    ax.set_xlabel("height [m]"); ax.set_ylabel("pdf")
    ax.set_title(f"Gaussianity (excess kurtosis={kurtosis(hC):+.3f})"); ax.legend()
    fig.tight_layout(); fig.savefig("fig4_gaussianity.png", dpi=110)
    print("[saved] fig4_gaussianity.png")

    # Fig 5: convergence of the COVARIANCE-kernel error vs M (Rahimi-Recht).
    # For each sample set we form the kernel the model reproduces in expectation
    # over phases:  C_model(r) = sum_j (a_j^2/2) cos(k_j . r),  and compare to the
    # analytic target C(r) = int F(k) J0(kr) dk.  Pure estimator error, no grid.
    Ms = [8, 16, 32, 64, 128, 256, 512, 1024]
    nseed = 24
    r = np.linspace(0.0, 150.0, 200)
    rvec = np.stack([r, np.zeros_like(r)], 1)            # measure along x
    Ctgt = analytic_autocov(r, U)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme, col in [("A", "tab:red"), ("B", "tab:blue"), ("C", "tab:green")]:
        err = []
        for Mv in Ms:
            e = []
            for sd in range(nseed):
                w = sample_waves(Mv, U, scheme, rng=np.random.default_rng(1000 + sd))
                phase_arg = rvec @ w["k"].T                  # (R, M)
                Cmod = (0.5 * w["amp"]**2) * np.cos(phase_arg)
                Cmod = Cmod.sum(1)
                e.append(np.sqrt(np.mean((Cmod - Ctgt)**2)) / sigma2)
            err.append(np.mean(e))
        ax.loglog(Ms, err, "o-", color=col, label=f"Scheme {scheme}")
    ax.loglog(Ms, 0.7 * np.array(Ms, float)**-0.5, "k:", label="~M^{-1/2}")
    ax.set_xlabel("M (number of waves)")
    ax.set_ylabel("RMS kernel error  ||C_model - C_target|| / sigma^2")
    ax.set_title("Covariance-kernel convergence vs M (Rahimi-Recht)"); ax.legend()
    fig.tight_layout(); fig.savefig("fig5_convergence.png", dpi=110)
    print("[saved] fig5_convergence.png")

    print("\nDONE.")

if __name__ == "__main__":
    main()
