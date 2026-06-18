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

// Sea-of-Thieves style shading: blend a deep colour and a subsurface colour by sun
// direction and a wave-peak mask (the peak mask comes from the choppiness offset).
uniform vec3  deep_color  : source_color = vec3(0.01, 0.06, 0.13);
uniform vec3  sss_color   : source_color = vec3(0.04, 0.42, 0.46);  // subsurface teal
uniform vec3  foam_color  : source_color = vec3(0.92, 0.96, 0.98);
uniform vec3  sun_dir = vec3(0.0, 0.6, 0.8);   // direction toward the sun (world)
uniform vec3  sun_color : source_color = vec3(1.0, 0.9, 0.7);
uniform float sss_strength = 1.6;   // extra glow on backlit peaks
uniform float peak_scale = 0.5;     // maps choppiness offset to a 0..1 peak mask
uniform float sun_glare = 0.7;      // wide soft sun reflection
uniform float sun_glare_power = 40.0;

uniform vec3  lod_center = vec3(0.0);
uniform vec3  dbg_tint : source_color = vec3(0.0);
uniform float dbg_mix = 0.0;

uniform sampler2D foam_buf : filter_linear;   // persistent foam buffer (when foam_region > 0)
uniform float foam_region = 0.0;
uniform sampler2D foam_noise : repeat_enable, filter_linear_mipmap;  // tiling, mipmapped

// fine detail normal map (always on): sub-mesh ripple + breaks up reflection contours
uniform sampler2D detail_nmap : repeat_enable, filter_linear_mipmap;
uniform float nmap_scale1 = 0.06;
uniform float nmap_scale2 = 0.17;
uniform float nmap_strength = 0.22;

// high-frequency per-pixel normal detail (extra, experimental); enabled per LOD ring
uniform sampler2D detail_kop   : filter_nearest, repeat_disable;
uniform sampler2D detail_amp   : filter_nearest, repeat_disable;
uniform sampler2D detail_phase : filter_nearest, repeat_disable;
uniform int   detail_count = 0;    // 0 = off for this ring
uniform float detail_gain = 1.0;
uniform float detail_fade = 70.0;  // detail fades out by this distance

varying float v_foam;
varying float v_peak;     // choppiness-offset magnitude (wave peaks)
varying vec2  v_wpos;
varying vec3  v_wnormal;  // world-space geometric normal (for per-pixel detail)


void vertex() {
    vec2 X = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xz;
    float r = length(X - lod_center.xz);

    float h = 0.0;
    vec2 disp = vec2(0.0), slope = vec2(0.0);
    float Jxx = 0.0, Jxz = 0.0, Jzx = 0.0, Jzz = 0.0;
    for (int j = 0; j < lod_wave_count; j++) {
        vec3 kop  = texelFetch(wave_kop, ivec2(j, 0), 0).xyz;
        vec2 k    = kop.xy;
        float rkill = kop.z;
        float a   = texelFetch(wave_amp,   ivec2(j, 0), 0).r * steepness;
        float phi = texelFetch(wave_phase, ivec2(j, 0), 0).r;
        a *= 1.0 - smoothstep(fade_start * rkill, rkill, r);

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
    v_wnormal = n;

    float detJ = (1.0 + Jxx) * (1.0 + Jzz) - Jxz * Jzx;
    v_foam = clamp(foam_amount * (smoothstep(0.55, -0.05, detJ)
                   + 0.4 * smoothstep(1.2, 2.6, h)), 0.0, 1.0);
    v_peak = length(disp);     // larger horizontal offset == nearer a crest
    v_wpos = X;
}

void fragment() {
    vec3 V = normalize(VIEW);
    vec3 L = normalize((VIEW_MATRIX * vec4(sun_dir, 0.0)).xyz);

    // normal perturbation: fine detail normal map (always) breaks the smooth reflection
    // contours into sparkle; mipmaps fade it with distance so it never aliases.
    vec2 dn = vec2(0.0);
    vec2 m1 = texture(detail_nmap, v_wpos * nmap_scale1 + TIME * vec2(0.011, 0.014)).xy * 2.0 - 1.0;
    vec2 m2 = texture(detail_nmap, v_wpos * nmap_scale2 - TIME * vec2(0.017, 0.009)).xy * 2.0 - 1.0;
    dn += (m1 + m2) * nmap_strength;
    // optional extra per-pixel high-k modes (--detail), faded by distance
    if (detail_count > 0) {
        float r = length(v_wpos - lod_center.xz);
        float fade = detail_gain * (1.0 - smoothstep(detail_fade * 0.6, detail_fade, r));
        vec2 ds = vec2(0.0);
        for (int j = 0; j < detail_count; j++) {
            vec2 dk = texelFetch(detail_kop,   ivec2(j, 0), 0).xy;
            float da = texelFetch(detail_amp,   ivec2(j, 0), 0).r * fade;
            float dp = texelFetch(detail_phase, ivec2(j, 0), 0).r;
            ds += -da * sin(dot(dk, v_wpos) + dp) * dk;
        }
        dn += ds * 12.0;
    }
    vec3 wn = normalize(v_wnormal + vec3(-dn.x, 0.0, -dn.y));
    NORMAL = (VIEW_MATRIX * vec4(wn, 0.0)).xyz;
    float peak = clamp(v_peak * peak_scale, 0.0, 1.0);
    float toward_sun = clamp(dot(V, -L), 0.0, 1.0);

    // foam: fold/peak mask broken up by noise so it reads as bubbles
    float fn = 0.6 * texture(foam_noise, v_wpos * 0.06).r
             + 0.4 * texture(foam_noise, v_wpos * 0.21).r;
    float foam = clamp(v_foam * (0.55 + 0.6 * fn), 0.0, 1.0);
    if (foam_region > 0.0) {     // persistent foam buffer (camera-region)
        vec2 fuv = v_wpos / (2.0 * foam_region) + 0.5;
        if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0)
            foam = max(foam, texture(foam_buf, fuv).r * (0.7 + 0.6 * fn));
    }

    // SoT colour: deep <-> subsurface by sun direction and peak mask
    float scatter = clamp(peak * (0.35 + 0.65 * toward_sun), 0.0, 1.0);
    vec3 body = mix(deep_color, sss_color, scatter);
    ALBEDO = mix(body, foam_color, foam);
    ALBEDO = mix(ALBEDO, dbg_tint, dbg_mix);

    ROUGHNESS = mix(0.10, 0.9, foam);     // soft sky reflection (0.02 was mirror -> hard outlines)
    // specular AA, capped so crests and the horizon stay reflective (not matte)
    vec3 dnx = dFdx(NORMAL), dny = dFdy(NORMAL);
    float aa = clamp(0.4 * (dot(dnx, dnx) + dot(dny, dny)), 0.0, 0.12);
    ROUGHNESS = min(1.0, sqrt(ROUGHNESS * ROUGHNESS + aa));
    METALLIC  = 0.0;
    SPECULAR  = 0.6;

    // extra translucent glow on strongly backlit peaks
    EMISSION += sss_color * (sss_strength * scatter * toward_sun) * (1.0 - foam);
    // wide soft sun glare (area-light look)
    vec3 Rv = reflect(-V, NORMAL);
    float sd = clamp(dot(Rv, L), 0.0, 1.0);
    float glare = pow(sd, sun_glare_power) * sun_glare;   // wide soft glitter path
    float glint = pow(sd, 900.0) * 3.0;                    // sharp bright sun
    EMISSION += sun_color * (glare + glint) * (1.0 - foam);
}
