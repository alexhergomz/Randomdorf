#!/usr/bin/env python3
"""Real GPU comparison on the RTX 4050: cuFFT Tessendorf vs optimized CUDA RFF.
Both measured with CUDA events (true GPU time) + peak VRAM."""
import torch, math
dev = "cuda"
G = 9.81
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))

def gpu_time(fn, iters=100, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for i in range(iters): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters, torch.cuda.max_memory_allocated()/1024/1024

# ---------------- Tessendorf via cuFFT ----------------
def make_fft(N, L=256.0):
    fx = torch.fft.fftfreq(N, d=L/N, device=dev)*2*math.pi
    fz = torch.fft.rfftfreq(N, d=L/N, device=dev)*2*math.pi
    KX, KZ = torch.meshgrid(fx, fz, indexing="ij"); K = torch.hypot(KX, KZ).clamp_min(1e-6)
    omega = torch.sqrt(G*K)
    h0 = (torch.randn(K.shape, device=dev)+1j*torch.randn(K.shape, device=dev)).to(torch.complex64)
    h0 *= torch.sqrt((torch.exp(-1/(K*50)**2)/K**2/2)).to(torch.complex64)
    mx = (1j*KX/K).to(torch.complex64); mz = (1j*KZ/K).to(torch.complex64)
    sx = (1j*KX).to(torch.complex64); sz = (1j*KZ).to(torch.complex64)
    omega = omega.to(torch.float32)
    def frame(t=0):
        ht = h0*torch.exp(1j*omega*float(t)*0.1).to(torch.complex64)
        # packed: 3 complex->real iFFTs giving 5 fields (h+dx, dz+sx, sz)
        a = torch.fft.irfft2(ht + 1j*mx*ht, s=(N, N))
        b = torch.fft.irfft2(mz*ht + 1j*sx*ht, s=(N, N))
        c = torch.fft.irfft2(sz*ht, s=(N, N))
        return a, b, c
    return frame

# ---------------- optimized RFF (CUDA) ----------------
def make_rff(M, P, L=256.0):
    km = torch.logspace(math.log10(0.05), math.log10(3.0), M, device=dev)
    th = torch.rand(M, device=dev)*2*math.pi
    K = torch.stack([km*torch.cos(th), km*torch.sin(th)]).to(torch.float32)  # (2,M)
    om = torch.sqrt(G*km).to(torch.float32); ph = (torch.rand(M, device=dev)*2*math.pi).to(torch.float32)
    am = (torch.ones(M, device=dev)/math.sqrt(M)).to(torch.float32)
    wkx = am*K[0]; wkz = am*K[1]
    pts = (torch.rand(P, 2, device=dev)*L).to(torch.float32)                 # P render points
    def frame(t=0):
        TH = pts @ K + (ph - om*float(t)*0.1)            # (P,M)
        C = torch.cos(TH); S = torch.sin(TH)
        h = C @ am; sx = -(S @ wkx); sz = -(S @ wkz); dx = -(S @ wkx); dz = -(S @ wkz)
        return h, sx, sz, dx, dz
    return frame

print("\n=== Tessendorf (cuFFT, packed 3-transform), per cascade ===")
for N in [256, 512]:
    ms, vram = gpu_time(make_fft(N))
    print(f"  N={N:4d}: {ms:.3f} ms/frame  | x3 cascades = {ms*3:.3f} ms  | VRAM {vram:.1f} MB")

print("\n=== Optimized RFF (CUDA parallel trig), at P render points ===")
for M in [48, 64, 96]:
    for P in [77_000, 150_000, 360_000, 1_000_000]:
        ms, vram = gpu_time(make_rff(M, P))
        print(f"  M={M:3d}  P={P:>9}: {ms:.3f} ms/frame  | VRAM {vram:.1f} MB")
    print()

print("Notes: FFT cost is FIXED regardless of how many points sample the tile (then ~free")
print("texture fetches). RFF cost = M x P. With LOD-omission the effective M per far point")
print("drops (~50 avg in our clipmap), so real RFF scene cost is below the M=64 rows.")
