shader_type spatial;
render_mode cull_back, specular_schlick_ggx;

// Per-wave data, M x 1 textures, sorted by |k| ascending:
//   wave_kop   = (kx, kz, r_kill, _)        r_kill = distance the wave stops being resolvable
//   wave_amp   = amplitude
//   wave_phase = wrap(base_phase - omega*t), refreshed per frame on the CPU
uniform sampler2D wave_kop   : filter_nearest, repeat_disable;
uniform sampler2D wave_amp   : filter_nearest, repeat_disable;
uniform sampler2D wave_phase : filter_nearest, repeat_disable;

uniform int   lod_wave_count = 96; // this ring only sums the modes its grid can resolve
uniform float fade_start = 0.6;    // each wave fades over [fade_start*r_kill, r_kill]
uniform float steepness = 2.0;
uniform float choppiness = 1.9;
uniform float foam_amount = 1.0;

uniform vec3 shallow_color : source_color = vec3(0.05, 0.33, 0.42);
uniform vec3 deep_color    : source_color = vec3(0.01, 0.06, 0.12);
uniform vec3 foam_color    : source_color = vec3(0.92, 0.96, 0.98);

uniform vec3  lod_center = vec3(0.0);          // LOD/fade origin (decoupled from camera)
uniform vec3  dbg_tint : source_color = vec3(0.0);  // per-ring debug color
uniform float dbg_mix = 0.0;                    // 0 = normal, >0 = show LOD rings

varying float v_foam;
varying float v_height;

void vertex() {
    vec2 X = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xz;
    vec2 cam = lod_center.xz;
    float r = length(X - cam);

    float h = 0.0;
    vec2 disp = vec2(0.0), slope = vec2(0.0);
    float Jxx = 0.0, Jxz = 0.0, Jzx = 0.0, Jzz = 0.0;
    for (int j = 0; j < lod_wave_count; j++) {
        vec3 kop  = texelFetch(wave_kop, ivec2(j, 0), 0).xyz;
        vec2 k    = kop.xy;
        float rkill = kop.z;
        float a   = texelFetch(wave_amp,   ivec2(j, 0), 0).r * steepness;
        float phi = texelFetch(wave_phase, ivec2(j, 0), 0).r;

        // fade out before the wave would alias; depends only on (k, r), so LOD seams match
        float fade = 1.0 - smoothstep(fade_start * rkill, rkill, r);
        a *= fade;

        float th = dot(k, X) + phi;
        float c = cos(th), s = sin(th);
        float kl = max(length(k), 1e-6);
        vec2 kd = k / kl;
        h     += a * c;
        disp  -= choppiness * kd * (a * s);
        slope += -a * s * k;
        float g = -choppiness * a * c;
        Jxx += g * kd.x * k.x;  Jxz += g * kd.x * k.y;
        Jzx += g * kd.y * k.x;  Jzz += g * kd.y * k.y;
    }
    VERTEX.x += disp.x;
    VERTEX.y += h;
    VERTEX.z += disp.y;

    vec3 Tx = vec3(1.0 + Jxx, slope.x, Jzx);
    vec3 Tz = vec3(Jxz,       slope.y, 1.0 + Jzz);
    vec3 n = normalize(cross(Tz, Tx));
    if (n.y < 0.0) n = -n;
    NORMAL = n;

    float detJ = (1.0 + Jxx) * (1.0 + Jzz) - Jxz * Jzx;
    v_foam = clamp(foam_amount * (smoothstep(0.55, -0.05, detJ)
                   + 0.4 * smoothstep(1.2, 2.6, h)), 0.0, 1.0);
    v_height = h;
}

void fragment() {
    float depth_t = clamp(v_height * 0.25 + 0.5, 0.0, 1.0);
    vec3 base = mix(deep_color, shallow_color, depth_t);
    ALBEDO    = mix(base, foam_color, v_foam);
    ALBEDO    = mix(ALBEDO, dbg_tint, dbg_mix);   // LOD-ring debug overlay
    ROUGHNESS = mix(0.04, 0.92, v_foam);
    METALLIC  = 0.0;
    SPECULAR  = 0.7;
}
