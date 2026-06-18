shader_type canvas_item;
render_mode blend_add, unshaded;

// Renders the current breaking-foam mask (Jacobian fold) over a fixed world window
// into the foam accumulation buffer. Added each frame; a separate decay rect (multiply)
// fades the buffer, so together they form a leaky temporal integrator (EMA) for foam.
uniform sampler2D wave_kop   : filter_nearest;
uniform sampler2D wave_amp   : filter_nearest;
uniform sampler2D wave_phase : filter_nearest;   // updated per frame on the CPU
uniform int   wave_count = 64;
uniform float steepness = 2.0;
uniform float choppiness = 1.5;
uniform float region = 256.0;        // half-extent of the world window (m)
uniform float source_gain = 0.10;    // per-frame foam added where breaking

void fragment() {
    vec2 X = (UV * 2.0 - 1.0) * region;     // buffer texel -> world XZ
    float Jxx = 0.0, Jxz = 0.0, Jzx = 0.0, Jzz = 0.0;
    for (int j = 0; j < wave_count; j++) {
        vec2 k    = texelFetch(wave_kop,   ivec2(j, 0), 0).xy;
        float a   = texelFetch(wave_amp,   ivec2(j, 0), 0).r * steepness;
        float phi = texelFetch(wave_phase, ivec2(j, 0), 0).r;
        float th = dot(k, X) + phi;
        float c = cos(th);
        float kl = max(length(k), 1e-6);
        vec2 kd = k / kl;
        float g = -choppiness * a * c;
        Jxx += g * kd.x * k.x;  Jxz += g * kd.x * k.y;
        Jzx += g * kd.y * k.x;  Jzz += g * kd.y * k.y;
    }
    float detJ = (1.0 + Jxx) * (1.0 + Jzz) - Jxz * Jzx;
    float src = smoothstep(0.25, -0.1, detJ) * source_gain;   // only near-breaking folds
    COLOR = vec4(src, src, src, 1.0);
}
