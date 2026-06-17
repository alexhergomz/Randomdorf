# Randomdorf

Real-valued ocean waves with Random Fourier Features instead of an FFT.

> **WIP / experiment.** Built to see how far a Tessendorf-style ocean gets as a direct
> sum of spectrally-sampled waves — no FFT, no displacement textures — and where that
> actually pays off against the usual FFT pipeline. Not production-ready.

![ocean](media/ocean.gif)

## Idea

The standard Tessendorf ocean draws a Gaussian height field by inverse-FFT of an
oceanographic spectrum (Phillips / Pierson–Moskowitz / JONSWAP). That field is itself a sum
of sinusoids, so instead of transforming a full grid you can **sample `M` wave components
straight from the spectrum and add them per vertex**. Pick the modes from the spectrum
(Wiener–Khinchin / random-phase model — the "RFF" view from kernel methods) and the
statistics match. The whole thing is one spatial shader: vertex displacement + analytic
normals, no compute, no ping-pong textures.

`analysis/verify_rff.py` checks it against the spectrum: variance = σ², autocovariance =
∫F(k)·J₀(kr)dk, Gaussian heights, M^(−1/2) convergence.

## LOD

Because it's a sum of components, level-of-detail just **omits** the modes a tile's grid
can't resolve (Nyquist) — no extra cost, no cracks (each mode fades with distance,
identically across tiles, so boundaries line up). Grid resolution scales with the render
resolution. Rings coloured by level:

![lod](media/lodview.gif)

## RFF vs FFT — honest

Measured on an RTX 4050 (`analysis/bench_gpu.py`):

| | FFT Tessendorf (3 cascades) | RFF (M=64, with LOD) |
|---|---|---|
| GPU / frame | ~0.55 ms | ~0.90 ms |
| Memory | ~5–12 MB (spectra + maps) | ~2 KB (wave table) |
| Spectral detail | thousands of modes | 64 modes |
| Normals | extra transforms | analytic, free |
| Query any point (buoyancy) | sample a texture | direct `h(x,z,t)` |
| Tiling | repeats; needs blending to hide | none needed |
| LOD | cascades / mip (awkward) | omit modes (free) |

**FFT is faster and richer, and that's fundamental.** One transform yields all N² modes in
O(N² log N) — `log R` work per output point versus `M` for RFF — so for a dense, detailed
field nothing beats it. Production FFT oceans are also a solved, well-tooled problem (see
Tessendorf's notes, and Ubisoft's [tiling-and-blending writeup](https://www.ubisoft.com/en-us/studio/laforge/news/5WHMK3tLGMGsqhxmWls1Jw/making-waves-in-ocean-surface-rendering-using-tiling-and-blending)
on the machinery needed just to hide tile repetition).

**RFF trades throughput for memory and simplicity:** ~1000× less memory, no
FFT/compute/texture pipeline (one shader), exact analytic normals, evaluate the surface at
any point (cheap buoyancy), no tiling repetition, and free Nyquist LOD.

So — FFT for a detailed AAA ocean; RFF when memory or pipeline simplicity matters
(mobile/web, many small water bodies, lots of physics queries) or when you want
repetition-free water with trivial LOD.

## Run

Godot 4.5+.

```
godot --path godot                 # ocean
godot --path godot -- --lodview    # coloured LOD rings
```

`godot/run.sh` forces the integrated GPU (a hybrid-laptop workaround; ignore otherwise).

## Notes / maybe later

- Foam is a cheap Jacobian-fold + crest term — no accumulation or advection.
- Steepness is boosted past the physical PM significant wave height for looks.
- Deep-water dispersion only.
- A small scrolling normal map could add sub-metre ripple the mesh can't carry.

Not trying to beat FFT on raw detail — that's its game.
