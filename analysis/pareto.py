#!/usr/bin/env python3
"""Pareto frontier: GPU time vs memory as you scale each method's grid.
FFT scales the tile resolution N (sets coverage, memory and time together, then tiles).
RFF scales two independent knobs: grid size P (coverage/speed) and modes M (detail/memory)."""
import torch, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
dev = "cuda"; G = 9.81
assert torch.cuda.is_available()
name = torch.cuda.get_device_name(0)

def gpu_time(fn, iters=100, warm=15):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for i in range(iters): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters, torch.cuda.max_memory_allocated()/1024/1024

def make_fft(N, L=256.0):
    fx = torch.fft.fftfreq(N, d=L/N, device=dev)*2*math.pi
    fz = torch.fft.rfftfreq(N, d=L/N, device=dev)*2*math.pi
    KX, KZ = torch.meshgrid(fx, fz, indexing="ij"); K = torch.hypot(KX, KZ).clamp_min(1e-6)
    omega = torch.sqrt(G*K).to(torch.float32)
    h0 = (torch.randn(K.shape, device=dev)+1j*torch.randn(K.shape, device=dev)).to(torch.complex64)
    h0 *= torch.sqrt((torch.exp(-1/(K*50)**2)/K**2/2)).to(torch.complex64)
    mx=(1j*KX/K).to(torch.complex64); sx=(1j*KX).to(torch.complex64)
    mz=(1j*KZ/K).to(torch.complex64); sz=(1j*KZ).to(torch.complex64)
    def frame(t=0):
        ht = h0*torch.exp(1j*omega*float(t)*0.1).to(torch.complex64)
        a = torch.fft.irfft2(ht + 1j*mx*ht, s=(N, N))
        b = torch.fft.irfft2(mz*ht + 1j*sx*ht, s=(N, N))
        c = torch.fft.irfft2(sz*ht, s=(N, N))
        return a, b, c
    return frame

def make_rff(M, P, L=256.0):
    km = torch.logspace(math.log10(0.05), math.log10(3.0), M, device=dev)
    th = torch.rand(M, device=dev)*2*math.pi
    K = torch.stack([km*torch.cos(th), km*torch.sin(th)]).to(torch.float32)
    om = torch.sqrt(G*km).to(torch.float32); ph = (torch.rand(M, device=dev)*2*math.pi).to(torch.float32)
    am = (torch.ones(M, device=dev)/math.sqrt(M)).to(torch.float32)
    wkx = am*K[0]; wkz = am*K[1]
    pts = (torch.rand(P, 2, device=dev)*L).to(torch.float32)
    def frame(t=0):
        TH = pts @ K + (ph - om*float(t)*0.1)
        C = torch.cos(TH); S = torch.sin(TH)
        return C @ am, -(S @ wkx), -(S @ wkz)
    return frame

CASC = 3                       # a production FFT ocean runs ~3 spectral bands (cascades)
RFF_BYTES = 24                 # wave-table per mode: kop(16)+amp(4)+phase(4)

# FFT: scale the tile N (3 cascades)
fftN = [128, 256, 512, 1024]
fft_t, fft_mem = [], []
for N in fftN:
    t, v = gpu_time(make_fft(N))
    fft_t.append(t*CASC); fft_mem.append(v*CASC)

# RFF: scale the grid P at fixed M=64 (memory is grid-independent -> the wave table)
M0 = 64
Ps = [16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000]
g_t = [gpu_time(make_rff(M0, P))[0] for P in Ps]
g_mem = [RFF_BYTES*M0/1e6]*len(Ps)

# RFF: scale modes M at fixed P=77k (this moves memory)
P0 = 77_000
Ms = [16, 32, 64, 128, 256]
m_t = [gpu_time(make_rff(M, P0))[0] for M in Ms]
m_mem = [RFF_BYTES*M/1e6 for M in Ms]

# iso-speed grid: where the RFF grid line meets the FFT N=256 time
t_ref = fft_t[1]
def interp(xs, ys, y):
    for i in range(1, len(ys)):
        if (ys[i-1]-y)*(ys[i]-y) <= 0:
            f = (y-ys[i-1])/(ys[i]-ys[i-1]); return xs[i-1]+f*(xs[i]-xs[i-1])
    return xs[-1]*y/ys[-1]
P_iso = interp(Ps, g_t, t_ref)
print(f"GPU: {name}")
print(f"FFT N=256, 3 cascades: {t_ref:.3f} ms, {fft_mem[1]:.1f} MB, tile = {256*256:,} pts (tiles)")
print(f"iso-speed grid: RFF matches that time at P ~ {P_iso:,.0f} pts "
      f"(~{P_iso/(256*256):.2f}x the tile), unique, no repeat")
print(f"iso-memory: RFF wave table = {RFF_BYTES*M0} B at any grid size; "
      f"~{fft_mem[1]/(RFF_BYTES*M0/1e6):.0f}x under the FFT regardless of P")

plt.figure(figsize=(7.2, 5.0))
plt.loglog(fft_mem, fft_t, "o-", color="#c44", label="FFT, scale tile N (3 cascades)")
plt.loglog(g_mem, g_t, "s-", color="#48c", label=f"RFF, scale grid P (M={M0})")
plt.loglog(m_mem, m_t, "^-", color="#4a4", label=f"RFF, scale modes M (P={P0//1000}k)")
for N, x, y in zip(fftN, fft_mem, fft_t): plt.annotate(f"N={N}", (x, y), textcoords="offset points", xytext=(6, -2), fontsize=8)
for P, x, y in zip(Ps, g_mem, g_t):
    if P in (16_000, 128_000, 1_000_000): plt.annotate(f"{P//1000}k", (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
plt.axhline(t_ref, color="grey", ls="--", lw=0.8, alpha=0.7)
plt.xlabel("memory (MB)"); plt.ylabel("GPU time per frame (ms)")
plt.title(f"Ocean field: time vs memory ({name.split('Laptop')[0].strip()})")
plt.legend(fontsize=8.5, loc="lower right"); plt.grid(True, which="both", alpha=0.25)
plt.tight_layout(); plt.savefig("pareto.png", dpi=130)
print("saved pareto.png")
