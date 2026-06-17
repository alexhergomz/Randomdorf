extends Node3D

# Real-valued Tessendorf ocean via Random Fourier Features.
# M waves sampled from a Pierson-Moskowitz spectrum, summed per-vertex on a
# geometry clipmap. Each LOD ring omits the modes its grid can't resolve (Nyquist);
# the grid resolution scales with the render resolution. Run with `-- --lodview`
# to colour the LOD rings.

const G := 9.81
const U := 12.0           # wind speed (m/s)
const M := 64             # wave components
const WIND_DIR := 0.6
const SPREAD := 2.0       # cos^2s directional spreading
const KMAX := 3.0         # shortest resolved wave ~2 m
const STEEPNESS := 2.0    # Gerstner amplitude gain (artistic; exceeds physical Hs)
const CHOP := 1.9         # horizontal sharpening
const LEVELS := 6         # clipmap rings
const RES := 4.0          # samples/wavelength to keep a mode in a ring
const FADE := 0.6         # mode fades over [FADE*r_kill, r_kill]
const CAM_FOV := 62.0     # vertical FOV (deg)
const TRI_PX := 20.0      # target on-screen triangle size

var _kx := PackedFloat32Array()
var _kz := PackedFloat32Array()
var _om := PackedFloat32Array()
var _ph := PackedFloat32Array()
var _am := PackedFloat32Array()
var _rkill := PackedFloat32Array()
var _sigma2 := 0.0
var _clip_cells := 128
var _c0 := 0.5
var _levels: Array = []
var _phase_img: Image
var _phase_tex: ImageTexture
var _phase_buf := PackedFloat32Array()
var _cam: Camera3D
var _buoy: Node3D
var _t := 0.0
var _focus := Vector3.ZERO
var _lodview := false

func _ready() -> void:
	_lodview = OS.get_cmdline_user_args().has("--lodview")
	get_viewport().msaa_3d = Viewport.MSAA_4X
	_compute_lod()
	_build_waves()
	_setup_world()
	_setup_ocean()
	_setup_buoy()
	print("RFF ocean: M=%d  Hs~%.1f m  cells=%d  reach=+-%.0f m"
		% [M, 4.0 * sqrt(_sigma2) * STEEPNESS, _clip_cells,
		   (_clip_cells / 2.0) * _c0 * pow(2.0, LEVELS - 1)])

func _process(dt: float) -> void:
	_t += dt
	_update_phases(_t)
	_move_camera(_t)
	_update_buoy(_t)

# --- Pierson-Moskowitz spectrum -> wavenumber form ---
func _pm(omega: float) -> float:
	if omega <= 1e-6:
		return 0.0
	var wp := 0.855 * G / U
	return 8.1e-3 * G * G / pow(omega, 5.0) * exp(-1.25 * pow(wp / omega, 4.0))

func _Fk(k: float) -> float:
	if k <= 1e-9:
		return 0.0
	return _pm(sqrt(G * k)) * 0.5 * sqrt(G / k)

# --- LOD grid resolution from viewport (constant on-screen triangle size) ---
func _compute_lod() -> void:
	var h := float(get_viewport().size.y)
	if h < 64.0:
		h = 1080.0
	var f := (h * 0.5) / tan(deg_to_rad(CAM_FOV) * 0.5)
	_c0 = TAU / (RES * KMAX)
	var n := int(round(3.0 * f / TRI_PX))
	_clip_cells = clampi(n - (n % 2), 64, 256)

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
		var d := pow(cos(th * 0.5), 2.0 * SPREAD)
		if i > 0:
			dacc += 0.5 * (d + dprev) * dth
		th_tab[i] = th
		dcdf[i] = dacc

	var kmin := 0.02
	var ratio: float = pow(KMAX / kmin, 1.0 / float(M))
	var e0 := (_clip_cells / 2.0) * _c0
	var waves: Array = []
	for j in range(M):
		var ke0: float = kmin * pow(ratio, float(j))
		var ke1: float = kmin * pow(ratio, float(j + 1))
		var kmag: float = sqrt(ke0 * ke1)
		var amp: float = sqrt(2.0 * _Fk(kmag) * (ke1 - ke0))
		var theta := WIND_DIR + _invert(dcdf, th_tab, rng.randf() * dacc)
		var l_max := 0
		while l_max < LEVELS - 1 and _c0 * pow(2.0, l_max + 1) <= TAU / (RES * kmag):
			l_max += 1
		waves.append({"k": kmag, "kx": kmag * cos(theta), "kz": kmag * sin(theta),
			"om": sqrt(G * kmag), "ph": rng.randf() * TAU, "am": amp,
			"rk": e0 * pow(2.0, l_max)})
	waves.sort_custom(func(a, b): return a.k < b.k)
	for w in waves:
		_kx.append(w.kx); _kz.append(w.kz); _om.append(w.om)
		_ph.append(w.ph); _am.append(w.am); _rkill.append(w.rk)

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
		var a := _am[j] * STEEPNESS
		var kl: float = max(sqrt(_kx[j] * _kx[j] + _kz[j] * _kz[j]), 1e-6)
		var kdx := _kx[j] / kl; var kdz := _kz[j] / kl
		slx += -a * s * _kx[j]; slz += -a * s * _kz[j]
		h += a * c
		var g := -CHOP * a * c
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
		mat.set_shader_parameter("steepness", STEEPNESS)
		mat.set_shader_parameter("choppiness", CHOP)
		mat.set_shader_parameter("dbg_tint", Color.from_hsv(float(l) / LEVELS * 0.8, 0.9, 1.0))
		mat.set_shader_parameter("dbg_mix", 0.5 if _lodview else 0.0)
		var mi := MeshInstance3D.new()
		mi.mesh = _make_clip_level(l)
		mi.material_override = mat
		var ext: float = (_clip_cells / 2.0) * cell
		mi.custom_aabb = AABB(Vector3(-ext, -20, -ext), Vector3(2 * ext, 40, 2 * ext))
		mi.extra_cull_margin = 30.0
		add_child(mi)
		_levels.append({"mi": mi, "mat": mat})

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
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-30.0, 40.0, 0.0)
	sun.light_energy = 1.4
	sun.light_color = Color(1.0, 0.95, 0.85)
	add_child(sun)
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

func _setup_buoy() -> void:
	_buoy = Node3D.new()
	add_child(_buoy)
	var raft := MeshInstance3D.new()
	var rm := BoxMesh.new(); rm.size = Vector3(5.0, 0.6, 5.0)
	raft.mesh = rm
	var rmat := StandardMaterial3D.new(); rmat.albedo_color = Color(0.55, 0.35, 0.18)
	raft.material_override = rmat
	_buoy.add_child(raft)
	var ball := MeshInstance3D.new()
	var bm := SphereMesh.new(); bm.radius = 1.0; bm.height = 2.0
	ball.mesh = bm; ball.position = Vector3(0, 1.3, 0)
	var bmat := StandardMaterial3D.new()
	bmat.albedo_color = Color(0.9, 0.15, 0.1)
	bmat.emission_enabled = true; bmat.emission = Color(0.5, 0.05, 0.03)
	ball.material_override = bmat
	_buoy.add_child(ball)

# phases wrapped on the CPU (float64) so the GPU never sees a growing argument
func _update_phases(t: float) -> void:
	for j in range(M):
		_phase_buf[j] = fposmod(_ph[j] - _om[j] * t, TAU)
	_phase_img.set_data(M, 1, false, Image.FORMAT_RF, _phase_buf.to_byte_array())
	_phase_tex.update(_phase_img)

func _move_camera(t: float) -> void:
	if _lodview:
		_focus = Vector3(170.0 * sin(0.28 * t), 0.0, 130.0 * sin(0.19 * t))
		_cam.position = Vector3(0, 560.0, 300.0)
		_cam.look_at(Vector3.ZERO, Vector3.UP)
	else:
		var ang := 0.1 * t + 0.6
		_cam.position = Vector3(90.0 * sin(ang), 22.0, 90.0 * cos(ang))
		_cam.look_at(Vector3(0, 1.0, 0), Vector3.UP)
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
