# Randomdorf

Real-valued ocean waves built from Random Fourier Features instead of an FFT, in Godot.

> WIP / experiment. It sums spectral wave components directly rather than running an FFT, to
> see how that compares with the usual FFT ocean and where it pays off. Still rough.

![ocean](media/ocean.gif)

A Pierson-Moskowitz / JONSWAP spectrum sampled as M wave modes and summed in one spatial
shader, with analytic normals, Nyquist LOD, cheap buoyancy at any point, and a stylized
shading pass with foam. Sea state and foam are command-line flags (see Run).

## Idea

The standard Tessendorf ocean builds a Gaussian height field by inverse-FFT of an
oceanographic spectrum (Phillips, Pierson-Moskowitz, JONSWAP). That field is itself a sum of
sinusoids, so you can sample M wave components straight from the spectrum and add them per
vertex instead of transforming a full grid. Pick the modes from the spectrum (Wiener-Khinchin
/ random-phase model, the "RFF" view from kernel methods) and the statistics match. The wave
field fits in one spatial shader that does the displacement and the analytic normals.

`analysis/verify_rff.py` checks it against the spectrum: variance equals sigma^2,
autocovariance equals the analytic integral of F(k)*J0(kr), heights are Gaussian, and the
error drops as M^(-1/2).

## LOD

Because it is a sum of components, a level of detail simply drops the modes a tile's grid is
too coarse to resolve (Nyquist). It costs nothing extra and the seams stay closed: every mode
fades with distance the same way on both sides of a boundary, so the tiles line up. Grid
resolution scales with the render resolution. Rings coloured by level:

![lod](media/lodview.gif)

## RFF vs FFT, measured

At equal unique coverage (65k distinct samples) on an RTX 4050 (`analysis/pareto.py`):

| | FFT Tessendorf (3 cascades) | RFF (M=64) |
|---|---|---|
| GPU / frame | ~0.3 ms | ~1.3 ms |
| Memory (65k samples) | ~8 MB | ~1.3 MB |
| Spectral detail | thousands of modes | 64 modes |
| Normals | extra transforms | analytic, free |
| Query any point (buoyancy) | sample a texture | direct h(x,z,t) |
| Coverage | one patch, tiled (repeats) | unique, no repeat |
| LOD | cascades / mip, awkward | drop modes, free |

The FFT pulls ahead as the field grows and carries far more detail, and that is fundamental.
One transform yields all N^2 modes in O(N^2 log N), which works out to log R per output point
against M for RFF, so for a dense detailed field it wins on speed outright. Production FFT
oceans are also a solved, well-tooled
problem (see Tessendorf's notes, and Ubisoft's
[tiling-and-blending writeup](https://www.ubisoft.com/en-us/studio/laforge/news/5WHMK3tLGMGsqhxmWls1Jw/making-waves-in-ocean-surface-rendering-using-tiling-and-blending)
on the machinery used just to hide tile repetition).

RFF trades that throughput for a tiny stored footprint and a simple pipeline: an M-mode table
with the field streamed per vertex rather than a stored grid spectrum and maps (the Scaling
section has the full memory tradeoff), a single shader, exact analytic normals, the surface
value at any point for cheap buoyancy, coverage that stays seamless, and Nyquist LOD for free.

Use FFT for a detailed AAA ocean. Reach for RFF when memory or pipeline simplicity matter
(mobile, web, many small water bodies, lots of physics queries), or when you want seamless
coverage with cheap LOD.

Those numbers are the wave field alone. The full Godot scene shown here, with the shading
pass, foam, detail normal map, and reflection, measures about 2 to 3 ms at 720p on the RTX
4050; that cost applies on top of either wave method.

### Scaling to match the FFT

Both methods are sums of spectral modes. The FFT keeps one amplitude per grid wavenumber, so
its spectrum is O(grid), and it stores the field as textures, then tiles that field over any
area. RFF keeps M modes (a few KB) and evaluates the surface per vertex, so its coverage is
unique with no repeat. Producing P points is O(P) output for either method, so total memory
grows with the grid for both. Scaling the RFF grid to the FFT reference on the 4050
(`analysis/pareto.py`, warm-clock medians since a laptop GPU boosts cold then settles;
absolute times still jitter, so read the shape):

![pareto](media/pareto.png)

- Equal unique coverage (the fair axis): for the same number of distinct samples, RFF uses
  about 6x less memory across the range. On time it is faster at small fields (tens of
  thousands of points), about 4x slower at 65k, and about 10x at 1M, since RFF is linear in
  points while the FFT is N log N.
- Match speed: at the FFT's ~0.3 ms for three cascades (65k unique points), RFF does about
  32k unique points using under 1 MB against the FFT's ~8 MB, so about 12x lighter.
- Match memory: at the FFT's ~8 MB, RFF covers about 390k unique points, but at ~10 ms, about
  30x slower.

RFF is not grid-independent: its mode table is tiny, but the rest of its memory is the output
it produces, the same O(grid) the FFT pays. Tiling gets no credit here: the FFT can repeat its
one patch to fill more area, but that repetition is not physical and the pipeline spends real
effort hiding it (the Ubisoft writeup above), while RFF coverage is unique by construction. So
the honest summary is a memory-for-time trade at equal unique detail, about 7x lighter and,
beyond small fields, a few times slower.

## Run

Godot 4.5+.

```
godot --path godot                    # ocean
godot --path godot -- --sea=storm     # calm, moderate, rough, storm, swell
godot --path godot -- --foam-buffer   # persistent foam accumulation
godot --path godot -- --lodview       # coloured LOD rings
```

`godot/run.sh` is a convenience launcher. On a hybrid laptop, render on the discrete GPU (for
example `prime-select nvidia`).

## Notes

- Shading is a stylized pass after the Sea of Thieves look: a deep colour blended with a
  subsurface colour by sun direction and a wave-peak mask, soft sky reflection, a sharp sun
  glint, and a tiling mipmapped normal map for the sub-metre ripple the mesh is too coarse to
  carry.
- Foam is a Jacobian-fold-plus-crest mask at breaking crests. `--foam-buffer` adds an optional
  persistent accumulation buffer (a decaying SubViewport) over a world-fixed region.
- Steepness is pushed past the physical PM significant wave height for looks.
- Deep-water dispersion only.

## Maybe later

- Shallow-water dispersion (tanh(kd)) for coastlines.
- A camera-following foam buffer that advects, instead of the current world-fixed region.
- Spray or whitecap particles on the steepest breaks.
