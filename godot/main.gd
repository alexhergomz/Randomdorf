extends Node3D

# Real-valued Tessendorf ocean via Random Fourier Features.
# M waves sampled from a Pierson-Moskowitz spectrum, summed per-vertex on a
# geometry clipmap. Each LOD ring omits the modes its grid can't resolve (Nyquist);
# the grid resolution scales with the render resolution. Run with `-- --lodview`
# to colour the LOD rings.

const G := 9.81
const M := 64             # wave components
const WIND_DIR := 0.6
const KMAX := 3.0         # shortest resolved wave ~2 m

# sea state (override with --sea=NAME or --wind= --chop= --gamma=)
var _U := 11.0            # wind speed (m/s), sets spectrum scale
var _gamma := 2.0         # JONSWAP peak enhancement (1 = Pierson-Moskowitz)
var _spread := 2.0        # cos^2s directional spread (higher = narrower)
var _chop := 1.4          # Gerstner choppiness lambda; this is the peakiness control
var _steep := 1.0         # physical amplitudes (1.0); raise only for stylised exaggeration
const LEVELS := 6         # clipmap rings
const RES := 4.0          # samples/wavelength (4 = smooth; 2 = Nyquist floor, blocky)
const FADE := 0.6         # mode fades over [FADE*r_kill, r_kill]
const CAM_FOV := 62.0     # vertical FOV (deg)
const TRI_PX := 8.0       # target on-screen triangle size (smaller = finer mesh)

var _kx := PackedFloat32Array()
var _kz := PackedFloat32Array()
var _om := PackedFloat32Array()
var _ph := PackedFloat32Array()
var _am := PackedFloat32Array()
var _rkill := PackedFloat32Array()
var _sigma2 := 0.0

# high-frequency detail modes (per-pixel normal only, k beyond KMAX)
const M_DETAIL := 24
const KDETAIL := 12.0
var _detail_rings := 0       # how many of the finest rings get per-pixel detail
var _measure := false
var _dkx := PackedFloat32Array()
var _dkz := PackedFloat32Array()
var _dom := PackedFloat32Array()
var _dph := PackedFloat32Array()
var _dam := PackedFloat32Array()
var _dphase_buf := PackedFloat32Array()
var _dphase_img: Image
var _dphase_tex: ImageTexture
var _dkop_tex: ImageTexture
var _damp_tex: ImageTexture
var _mframes := 0
var _msum := 0.0
var _clip_cells := 128
var _c0 := 0.5
var _levels: Array = []
var _phase_img: Image
var _phase_tex: ImageTexture
var _phase_buf := PackedFloat32Array()
var _cam: Camera3D
var _sun: DirectionalLight3D
var _buoy: Node3D
var _t := 0.0
var _focus := Vector3.ZERO
var _lodview := false
var _foam_buffer := false
var _foam_vp: SubViewport
var _noise_tex: NoiseTexture2D
var _nmap_tex: NoiseTexture2D
var _shots := false
var _shots_dir := "user://shots"
var _shot_done := false
const FOAM_REGION := 160.0   # foam buffer covers +-this many metres around origin
const FOAM_RES := 640

func _ready() -> void:
	_parse_args(OS.get_cmdline_user_args())
	get_viewport().msaa_3d = Viewport.MSAA_4X
	if _measure:
		RenderingServer.viewport_set_measure_render_time(get_viewport().get_viewport_rid(), true)
	_compute_lod()
	_build_waves()
	_build_detail()
	_setup_world()
	_setup_ocean()
	_setup_buoy()
	print("RFF ocean: M=%d  Hs~%.1f m  chop=%.2f  cells=%d  reach=+-%.0f m"
		% [M, 4.0 * sqrt(_sigma2) * _steep, _chop, _clip_cells,
		   (_clip_cells / 2.0) * _c0 * pow(2.0, LEVELS - 1)])

func _process(dt: float) -> void:
	_t += dt
	_update_phases(_t)
	_move_camera(_t)
	_update_buoy(_t)
	if _measure:
		_mframes += 1
		if _mframes > 40:                          # warm up, then average GPU time
			var rid := get_viewport().get_viewport_rid()
			_msum += RenderingServer.viewport_get_measured_render_time_gpu(rid)
		if _mframes == 200:
			print("MEASURE detail_rings=%d : GPU %.3f ms (avg over 160 frames) at %dx%d"
				% [_detail_rings, _msum / 160.0, get_viewport().size.x, get_viewport().size.y])
			get_tree().quit()
		return
	if _shots and not _shot_done and _t > 2.0:   # let waves + foam buffer settle, then grab
		_shot_done = true
		DirAccess.make_dir_recursive_absolute(_shots_dir)
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(_shots_dir + "/view.png")
		if _foam_vp != null:
			_foam_vp.get_texture().get_image().save_png(_shots_dir + "/foam.png")
		await get_tree().create_timer(0.1).timeout
		get_tree().quit()

# --- args / sea state ---
func _parse_args(args: PackedStringArray) -> void:
	_lodview = args.has("--lodview")
	_foam_buffer = args.has("--foam-buffer")
	_measure = args.has("--measure")
	for a in args:
		if a.begins_with("--shots="):
			_shots = true
			_shots_dir = a.substr(8)
		elif a.begins_with("--detail="):
			_detail_rings = int(a.substr(9))
	for a in args:
		if a.begins_with("--sea="):
			_set_sea(a.substr(6))
		elif a.begins_with("--wind="):
			_U = float(a.substr(7))
		elif a.begins_with("--chop="):
			_chop = float(a.substr(7))
		elif a.begins_with("--gamma="):
			_gamma = float(a.substr(8))
		elif a.begins_with("--steep="):
			_steep = float(a.substr(8))

func _set_sea(name: String) -> void:
	# [wind, gamma, chop, spread, steep]
	var p := {
		"calm":     [6.0, 1.0, 0.9, 4.0, 1.0],
		"moderate": [11.0, 2.0, 1.4, 2.0, 1.0],
		"rough":    [17.0, 3.3, 1.7, 1.5, 1.0],
		"storm":    [9.0, 4.0, 2.5, 1.0, 4.0],     # stylised: short steep walls, towering
		"swell":    [14.0, 1.0, 0.7, 8.0, 1.0],
	}
	if p.has(name):
		var v: Array = p[name]
		_U = v[0]; _gamma = v[1]; _chop = v[2]; _spread = v[3]; _steep = v[4]

# JONSWAP spectrum (gamma = 1 reduces to Pierson-Moskowitz)
func _spectrum(omega: float) -> float:
	if omega <= 1e-6:
		return 0.0
	var wp := 0.855 * G / _U
	var pm := 8.1e-3 * G * G / pow(omega, 5.0) * exp(-1.25 * pow(wp / omega, 4.0))
	var sg := 0.07 if omega <= wp else 0.09
	var r := exp(-pow(omega - wp, 2.0) / (2.0 * sg * sg * wp * wp))
	return pm * pow(_gamma, r) * (1.0 - 0.287 * log(_gamma))

func _Fk(k: float) -> float:
	if k <= 1e-9:
		return 0.0
	return _spectrum(sqrt(G * k)) * 0.5 * sqrt(G / k)

# --- LOD grid resolution from viewport (constant on-screen triangle size) ---
func _compute_lod() -> void:
	var h := float(get_viewport().size.y)
	if h < 64.0:
		h = 1080.0
	var f := (h * 0.5) / tan(deg_to_rad(CAM_FOV) * 0.5)
	_c0 = TAU / (RES * KMAX)
	var n := int(round(3.0 * f / TRI_PX))
	_clip_cells = clampi(n - (n % 2), 64, 320)

# --- sample M waves: log-spaced k, amplitude from the spectrum, random phase/dir ---
func _build_waves() -> void:
	var nbin := 4096
	var dk := KMAX / float(nbin - 1)
	var prev := 0.0
	_sigma2 = 0.0
	for i in range(nbin):
		var fk := _Fk(max(1e-6, float(i) * dk))
		if i > 0:
			_sigma2 += 0.5 * (fk + prev) * dk
		prev = fk

	var rng := RandomNumberGenerator.new()
	rng.seed = 20260614
	var nd := 2048
	var th_tab := PackedFloat32Array(); th_tab.resize(nd)
	var dcdf := PackedFloat32Array(); dcdf.resize(nd)
	var dacc := 0.0
	var dth := TAU / float(nd - 1)
	var dprev := 0.0
	for i in range(nd):
		var th := -PI + float(i) * dth
		var d := pow(cos(th * 0.5), 2.0 * _spread)
		if i > 0:
			dacc += 0.5 * (d + dprev) * dth
		th_tab[i] = th
		dcdf[i] = dacc

	# low-k cutoff tracks the spectral peak so the energetic shoulder is sampled
	var kp: float = pow(0.855 * G / _U, 2.0) / G
	var kmin: float = clamp(kp * 0.15, 0.004, 0.05)
	var ratio: float = pow(KMAX / kmin, 1.0 / float(M))
	var e0 := (_clip_cells / 2.0) * _c0
	var waves: Array = []
	var var_sum := 0.0
	for j in range(M):
		var ke0: float = kmin * pow(ratio, float(j))
		var ke1: float = kmin * pow(ratio, float(j + 1))
		var kmag: float = sqrt(ke0 * ke1)
		var amp: float = sqrt(2.0 * _Fk(kmag) * (ke1 - ke0))
		# deterministic low-discrepancy direction (golden ratio): even, k-decorrelated, no lumpiness
		var u: float = fposmod(float(j) * 0.6180339887498949, 1.0)
		var theta := WIND_DIR + _invert(dcdf, th_tab, u * dacc)
		var l_max := 0
		while l_max < LEVELS - 1 and _c0 * pow(2.0, l_max + 1) <= TAU / (RES * kmag):
			l_max += 1
		var_sum += 0.5 * amp * amp
		waves.append({"k": kmag, "kx": kmag * cos(theta), "kz": kmag * sin(theta),
			"om": sqrt(G * kmag), "ph": rng.randf() * TAU, "am": amp,
			"rk": e0 * pow(2.0, l_max)})
	# rescale so the sampled set carries exactly sigma^2 (reported Hs == rendered Hs)
	var scale: float = sqrt(_sigma2 / max(var_sum, 1e-9))
	# clamp choppiness to the wave-set steepness so crests sharpen to the fold limit
	# but rarely invert (no chaotic self-crossing); the few that tip over become foam
	var sq := 0.0
	for w in waves:
		var ak: float = w.am * scale * w.k
		sq += ak * ak
	var chop_cap: float = 0.5 / max(_steep * sqrt(0.5 * sq), 1e-6)
	_chop = min(_chop, chop_cap)
	waves.sort_custom(func(a, b): return a.k < b.k)
	for w in waves:
		_kx.append(w.kx); _kz.append(w.kz); _om.append(w.om)
		_ph.append(w.ph); _am.append(w.am * scale); _rkill.append(w.rk)

# high-k modes for per-pixel normal detail (k in [KMAX, KDETAIL]); not displaced
func _build_detail() -> void:
	var ratio: float = pow(KDETAIL / KMAX, 1.0 / float(M_DETAIL))
	for j in range(M_DETAIL):
		var ke0: float = KMAX * pow(ratio, float(j))
		var ke1: float = KMAX * pow(ratio, float(j + 1))
		var kmag: float = sqrt(ke0 * ke1)
		var theta := WIND_DIR + (fposmod(float(j) * 0.6180339887, 1.0) * 2.0 - 1.0) * 1.2
		_dkx.append(kmag * cos(theta)); _dkz.append(kmag * sin(theta))
		_dom.append(sqrt(G * kmag))
		_dph.append(fposmod(float(j) * 0.7548776662, 1.0) * TAU)
		_dam.append(sqrt(2.0 * _Fk(kmag) * (ke1 - ke0)))
	_dphase_buf.resize(M_DETAIL)

func _invert(cdf: PackedFloat32Array, x_tab: PackedFloat32Array, target: float) -> float:
	var lo := 0
	var hi := cdf.size() - 1
	while lo < hi:
		var mid := (lo + hi) >> 1
		if cdf[mid] < target:
			lo = mid + 1
		else:
			hi = mid
	if lo == 0:
		return x_tab[0]
	var c0 := cdf[lo - 1]
	var c1 := cdf[lo]
	var f: float = 0.0 if c1 <= c0 else (target - c0) / (c1 - c0)
	return float(lerp(x_tab[lo - 1], x_tab[lo], f))

# --- CPU surface query (mirrors the shader) for buoyancy ---
func surface_at(x: float, z: float, t: float) -> Dictionary:
	var h := 0.0
	var slx := 0.0; var slz := 0.0
	var jxx := 0.0; var jxz := 0.0; var jzx := 0.0; var jzz := 0.0
	for j in range(M):
		var th := _kx[j] * x + _kz[j] * z - _om[j] * t + _ph[j]
		var c := cos(th); var s := sin(th)
		var a := _am[j] * _steep
		var kl: float = max(sqrt(_kx[j] * _kx[j] + _kz[j] * _kz[j]), 1e-6)
		var kdx := _kx[j] / kl; var kdz := _kz[j] / kl
		slx += -a * s * _kx[j]; slz += -a * s * _kz[j]
		h += a * c
		var g := -_chop * a * c
		jxx += g * kdx * _kx[j]; jxz += g * kdx * _kz[j]
		jzx += g * kdz * _kx[j]; jzz += g * kdz * _kz[j]
	var tx := Vector3(1.0 + jxx, slx, jzx)
	var tz := Vector3(jxz, slz, 1.0 + jzz)
	var n := tz.cross(tx).normalized()
	if n.y < 0.0:
		n = -n
	return {"h": h, "normal": n}

# --- scene ---
func _make_tex(data: PackedFloat32Array, fmt: int, w: int) -> ImageTexture:
	return ImageTexture.create_from_image(
		Image.create_from_data(w, 1, false, fmt, data.to_byte_array()))

func _setup_ocean() -> void:
	var kop := PackedFloat32Array()
	for j in range(M):
		kop.append(_kx[j]); kop.append(_kz[j]); kop.append(_rkill[j]); kop.append(0.0)
	var kop_tex := _make_tex(kop, Image.FORMAT_RGBAF, M)
	var amp_tex := _make_tex(_am, Image.FORMAT_RF, M)
	_phase_buf.resize(M)
	_phase_img = Image.create(M, 1, false, Image.FORMAT_RF)
	_phase_tex = ImageTexture.create_from_image(_phase_img)
	if _foam_buffer:
		_setup_foam_buffer(kop_tex, amp_tex)

	# tiling, mipmapped foam-breakup noise (replaces the procedural hash that went blocky far out)
	var fnl := FastNoiseLite.new()
	fnl.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	fnl.frequency = 0.035
	fnl.fractal_octaves = 4
	_noise_tex = NoiseTexture2D.new()
	_noise_tex.noise = fnl
	_noise_tex.seamless = true
	_noise_tex.generate_mipmaps = true
	_noise_tex.width = 256
	_noise_tex.height = 256

	# fine detail normal map (tiling, mipmapped) for sub-mesh ripple + reflection breakup
	var nnl := FastNoiseLite.new()
	nnl.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	nnl.frequency = 0.07
	nnl.fractal_octaves = 3
	_nmap_tex = NoiseTexture2D.new()
	_nmap_tex.noise = nnl
	_nmap_tex.seamless = true
	_nmap_tex.generate_mipmaps = true
	_nmap_tex.as_normal_map = true
	_nmap_tex.bump_strength = 8.0
	_nmap_tex.width = 256
	_nmap_tex.height = 256

	# detail mode textures (per-pixel high-frequency normal)
	var dkop := PackedFloat32Array()
	for j in range(M_DETAIL):
		dkop.append(_dkx[j]); dkop.append(_dkz[j]); dkop.append(0.0); dkop.append(0.0)
	_dkop_tex = _make_tex(dkop, Image.FORMAT_RGBAF, M_DETAIL)
	_damp_tex = _make_tex(_dam, Image.FORMAT_RF, M_DETAIL)
	_dphase_img = Image.create(M_DETAIL, 1, false, Image.FORMAT_RF)
	_dphase_tex = ImageTexture.create_from_image(_dphase_img)

	for l in range(LEVELS):
		var cell: float = _c0 * pow(2.0, l)
		var knyq: float = TAU / (RES * cell)
		var count := 0
		for j in range(M):
			if sqrt(_kx[j] * _kx[j] + _kz[j] * _kz[j]) <= knyq:
				count += 1
		count = maxi(count, 1)
		var mat := ShaderMaterial.new()
		mat.shader = load("res://ocean.gdshader")
		mat.set_shader_parameter("wave_kop", kop_tex)
		mat.set_shader_parameter("wave_amp", amp_tex)
		mat.set_shader_parameter("wave_phase", _phase_tex)
		mat.set_shader_parameter("lod_wave_count", count)
		mat.set_shader_parameter("fade_start", FADE)
		mat.set_shader_parameter("steepness", _steep)
		mat.set_shader_parameter("choppiness", _chop)
		mat.set_shader_parameter("sun_dir", _sun.global_transform.basis.z)
		mat.set_shader_parameter("dbg_tint", Color.from_hsv(float(l) / LEVELS * 0.8, 0.9, 1.0))
		mat.set_shader_parameter("dbg_mix", 0.5 if _lodview else 0.0)
		mat.set_shader_parameter("foam_noise", _noise_tex)
		mat.set_shader_parameter("detail_nmap", _nmap_tex)
		mat.set_shader_parameter("detail_kop", _dkop_tex)
		mat.set_shader_parameter("detail_amp", _damp_tex)
		mat.set_shader_parameter("detail_phase", _dphase_tex)
		mat.set_shader_parameter("detail_count", M_DETAIL if l < _detail_rings else 0)
		mat.set_shader_parameter("detail_gain", 4.0)
		mat.set_shader_parameter("detail_fade", 70.0)
		if _foam_buffer:
			mat.set_shader_parameter("foam_buf", _foam_vp.get_texture())
			mat.set_shader_parameter("foam_region", FOAM_REGION)
		var mi := MeshInstance3D.new()
		mi.mesh = _make_clip_level(l)
		mi.material_override = mat
		var ext: float = (_clip_cells / 2.0) * cell
		mi.custom_aabb = AABB(Vector3(-ext, -20, -ext), Vector3(2 * ext, 40, 2 * ext))
		mi.extra_cull_margin = 30.0
		add_child(mi)
		_levels.append({"mi": mi, "mat": mat})

# Foam accumulation buffer (--foam-buffer): a world-fixed SubViewport over +-FOAM_REGION.
# Each frame a decay rect multiplies the buffer down and the foam-source shader adds new
# breaking foam, so it is a leaky temporal integrator (EMA): foam lingers and disperses.
func _setup_foam_buffer(kop_tex: ImageTexture, amp_tex: ImageTexture) -> void:
	_foam_vp = SubViewport.new()
	_foam_vp.size = Vector2i(FOAM_RES, FOAM_RES)
	_foam_vp.render_target_clear_mode = SubViewport.CLEAR_MODE_ONCE
	_foam_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	_foam_vp.transparent_bg = true
	add_child(_foam_vp)
	var decay := ColorRect.new()
	decay.size = Vector2(FOAM_RES, FOAM_RES)
	decay.color = Color(0.93, 0.93, 0.93, 1.0)     # multiply buffer down each frame
	var dmat := CanvasItemMaterial.new()
	dmat.blend_mode = CanvasItemMaterial.BLEND_MODE_MUL
	decay.material = dmat
	_foam_vp.add_child(decay)
	var src := ColorRect.new()
	src.size = Vector2(FOAM_RES, FOAM_RES)
	var smat := ShaderMaterial.new()
	smat.shader = load("res://foam_source.gdshader")
	smat.set_shader_parameter("wave_kop", kop_tex)
	smat.set_shader_parameter("wave_amp", amp_tex)
	smat.set_shader_parameter("wave_phase", _phase_tex)
	smat.set_shader_parameter("wave_count", M)
	smat.set_shader_parameter("steepness", _steep)
	smat.set_shader_parameter("choppiness", _chop)
	smat.set_shader_parameter("region", FOAM_REGION)
	src.material = smat
	_foam_vp.add_child(src)

# clipmap ring: clip_cells^2 grid, central quarter removed (filled by the finer ring)
func _make_clip_level(level: int) -> ArrayMesh:
	var cell: float = _c0 * pow(2.0, level)
	var nc := _clip_cells
	var half := nc / 2
	var hole := nc / 4 if level > 0 else 0
	var verts := PackedVector3Array(); verts.resize((nc + 1) * (nc + 1))
	var idx := PackedInt32Array()
	for i in range(nc + 1):
		for j in range(nc + 1):
			verts[i * (nc + 1) + j] = Vector3((i - half) * cell, 0.0, (j - half) * cell)
	for i in range(nc):
		for j in range(nc):
			if level > 0 and i >= half - hole and i + 1 <= half + hole \
					and j >= half - hole and j + 1 <= half + hole:
				continue
			var a := i * (nc + 1) + j
			var b := (i + 1) * (nc + 1) + j
			var c := (i + 1) * (nc + 1) + (j + 1)
			var d := i * (nc + 1) + (j + 1)
			idx.append_array([a, b, c, a, c, d])
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = verts
	arr[Mesh.ARRAY_INDEX] = idx
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	return mesh

func _setup_world() -> void:
	_cam = Camera3D.new()
	_cam.fov = CAM_FOV
	_cam.far = 3000.0
	add_child(_cam)
	_sun = DirectionalLight3D.new()
	_sun.rotation_degrees = Vector3(-30.0, 40.0, 0.0)
	_sun.light_energy = 1.4
	_sun.light_color = Color(1.0, 0.95, 0.85)
	_sun.sky_mode = DirectionalLight3D.SKY_MODE_LIGHT_AND_SKY  # draw a sun disk to reflect
	add_child(_sun)
	var env := Environment.new()
	var sky := Sky.new()
	var skymat := ProceduralSkyMaterial.new()
	skymat.sky_top_color = Color(0.28, 0.52, 0.85)
	skymat.sky_horizon_color = Color(0.72, 0.83, 0.94)
	skymat.ground_bottom_color = Color(0.22, 0.34, 0.44)
	sky.sky_material = skymat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.ssr_enabled = true
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	# reflection probe: captures the sky (ocean excluded via cull_mask) for a proper
	# grazing-angle reflection where screen-space reflection misses
	var probe := ReflectionProbe.new()
	probe.size = Vector3(8000, 4000, 8000)
	probe.box_projection = false
	probe.interior = false
	probe.cull_mask = 0xFFFFE   # exclude layer 1 (ocean + buoy) -> sky only
	probe.update_mode = ReflectionProbe.UPDATE_ONCE
	add_child(probe)

func _setup_buoy() -> void:
	_buoy = Node3D.new()
	add_child(_buoy)
	var raft := MeshInstance3D.new()
	var rm := BoxMesh.new(); rm.size = Vector3(1.8, 0.35, 1.8)   # small boat, for scale
	raft.mesh = rm
	var rmat := StandardMaterial3D.new(); rmat.albedo_color = Color(0.55, 0.35, 0.18)
	raft.material_override = rmat
	_buoy.add_child(raft)
	var ball := MeshInstance3D.new()
	var bm := SphereMesh.new(); bm.radius = 0.4; bm.height = 0.8
	ball.mesh = bm; ball.position = Vector3(0, 0.5, 0)
	var bmat := StandardMaterial3D.new()
	bmat.albedo_color = Color(0.8, 0.18, 0.12); bmat.roughness = 0.6
	ball.material_override = bmat
	_buoy.add_child(ball)

# phases wrapped on the CPU (float64) so the GPU never sees a growing argument
func _update_phases(t: float) -> void:
	for j in range(M):
		_phase_buf[j] = fposmod(_ph[j] - _om[j] * t, TAU)
	_phase_img.set_data(M, 1, false, Image.FORMAT_RF, _phase_buf.to_byte_array())
	_phase_tex.update(_phase_img)
	if _dphase_tex != null:
		for j in range(M_DETAIL):
			_dphase_buf[j] = fposmod(_dph[j] - _dom[j] * t, TAU)
		_dphase_img.set_data(M_DETAIL, 1, false, Image.FORMAT_RF, _dphase_buf.to_byte_array())
		_dphase_tex.update(_dphase_img)

func _move_camera(t: float) -> void:
	if _lodview:
		_focus = Vector3(170.0 * sin(0.28 * t), 0.0, 130.0 * sin(0.19 * t))
		_cam.position = Vector3(0, 560.0, 300.0)
		_cam.look_at(Vector3.ZERO, Vector3.UP)
	else:
		# third person near the buoy: low and close so the waves loom, slow orbit
		var bx := 6.0 * sin(t * 0.15)
		var bz := 4.0 * cos(t * 0.13)
		var bpos := Vector3(bx, surface_at(bx, bz, t).h, bz)
		var yaw := 0.12 * t + 0.6
		_cam.position = bpos + Vector3(sin(yaw) * 11.0, 5.0, cos(yaw) * 11.0)
		_cam.look_at(bpos + Vector3(0.0, 0.8, 0.0), Vector3.UP)
		_focus = _cam.position
	var cx: float = round(_focus.x / _c0) * _c0
	var cz: float = round(_focus.z / _c0) * _c0
	var c := Vector3(cx, 0.0, cz)
	for lv in _levels:
		lv.mi.global_position = c
		lv.mat.set_shader_parameter("lod_center", c)

func _update_buoy(t: float) -> void:
	if _lodview:
		_buoy.visible = false
		return
	var bx := 6.0 * sin(t * 0.15)
	var bz := 4.0 * cos(t * 0.13)
	var s := surface_at(bx, bz, t)
	var yb: Vector3 = s.normal
	var xb := (Vector3.RIGHT - yb * Vector3.RIGHT.dot(yb)).normalized()
	_buoy.global_transform = Transform3D(Basis(xb, yb, xb.cross(yb)),
		Vector3(bx, s.h + 0.1, bz))
