"""F-01 slice 1 spike, 2D rig test (BUILD_LOG L-076, L-077): pixel maps that let one painted rest frame play
every guide frame.

Rounds 1–3 showed Muse Image won't copy guide poses (L-075), so only the rest frame (idle frame 0) is painted
per direction and the guide's own motion moves it. This module is the Blender half. For every frame it
ray-casts a 2 px grid to find the body surface point under each pixel, then finds where that point sits in
the rest frame:
  direct  the point was visible in the rest frame → copy the painted pixel from there
  mirror  it was hidden (the far leg in a side view) but its left/right twin was visible → copy the twin
  hidden  neither was visible → a hole, filled later from neighbours
It also keeps the forward map (rest pixel → frame pixel) from L-076, used only for clothing that sticks out
past the body. The body is inflated a little (Displace) so painted armour mostly falls inside the guide
silhouette.

Runs inside the Blender image (`modal_app.py::render_rig_guides`); numpy ships with bpy.
"""

import time

import numpy as np

from lote.guides.spike import CELL_H, CELL_W, _render_humanoid, _reset_scene

STEP = 2  # px between ray samples
INFLATE = 0.025  # m pushed out along normals: room for painted armour and cloth
VIS_EPS = 0.04  # m: a point counts as visible in the rest frame if it is this close to the front surface


def _world_verts(obj, dg) -> np.ndarray:
    """Evaluated (skinned, inflated) vertex positions in world space, shape (n, 3)."""
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    m = np.array(ev.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def _camera_frame(cam) -> dict:
    """Orthographic camera → what we need to go between world points and pixels."""
    cd = cam.data
    h = cd.ortho_scale  # sensor_fit VERTICAL: visible height in metres
    w = h * CELL_W / CELL_H
    mw = np.array(cam.matrix_world)
    return {"inv": np.linalg.inv(mw), "mw": mw, "w": w, "h": h,
            "sx": cd.shift_x * h, "sy": cd.shift_y * h}  # shift is a fraction of the larger side (height)


def _project(cf: dict, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points (n, 3) → pixel x, pixel y, depth (bigger = farther from the camera)."""
    loc = pts @ cf["inv"][:3, :3].T + cf["inv"][:3, 3]
    px = (loc[:, 0] - cf["sx"] + cf["w"] / 2) / cf["w"] * CELL_W
    py = (cf["h"] / 2 + cf["sy"] - loc[:, 1]) / cf["h"] * CELL_H
    return px, py, -loc[:, 2]


def _inflate(meshes) -> None:
    for m in meshes:
        md = m.modifiers.new("Inflate", "DISPLACE")  # after Armature: pushes the posed surface out
        md.strength, md.mid_level = INFLATE, 0.0


def _mirror_map(meshes) -> list[np.ndarray]:
    """Per mesh: index of each vertex's left/right twin in the bind pose. The symmetry axis is the local axis
    whose mirror gives the closest match."""
    from mathutils import Vector, kdtree

    out = []
    for m in meshes:
        co = np.array([v.co[:] for v in m.data.vertices])
        tree = kdtree.KDTree(len(co))
        for i, c in enumerate(co):
            tree.insert(Vector(c), i)
        tree.balance()
        best = None
        for axis in range(3):
            flip = co.copy()
            flip[:, axis] *= -1
            found = [tree.find(Vector(c)) for c in flip]
            err = float(np.mean([f[2] for f in found]))
            if best is None or err < best[0]:
                best = (err, np.array([f[1] for f in found], dtype=np.int64))
        out.append(best[1])
    return out


def _polys(meshes, dg) -> list[list[list[int]]]:
    out = []
    for m in meshes:
        me = m.evaluated_get(dg).to_mesh()
        out.append([list(p.vertices) for p in me.polygons])
        m.evaluated_get(dg).to_mesh_clear()
    return out


def _cast(scene, meshes, polys, dg) -> dict:
    """Ray-cast the STEP grid from the camera → per hit: grid cell, mesh, polygon vertex ids, point, depth."""
    from mathutils import Vector

    cf = _camera_frame(scene.camera)
    fwd = Vector((-cf["mw"][0, 2], -cf["mw"][1, 2], -cf["mw"][2, 2])).normalized()
    names = {m.name: k for k, m in enumerate(meshes)}
    gh, gw = -(-CELL_H // STEP), -(-CELL_W // STEP)
    rows = []
    for gy in range(gh):
        for gx in range(gw):
            lx = (gx * STEP + 0.5) / CELL_W * cf["w"] - cf["w"] / 2 + cf["sx"]
            ly = cf["h"] / 2 + cf["sy"] - (gy * STEP + 0.5) / CELL_H * cf["h"]
            o = cf["mw"] @ np.array([lx, ly, 0.0, 1.0])
            ok, loc, _n, face, obj, _m = scene.ray_cast(dg, Vector(o[:3]), fwd)
            if not ok or obj is None or obj.name not in names:
                continue
            k = names[obj.name]
            if 0 <= face < len(polys[k]):
                rows.append((gy, gx, k, polys[k][face], loc[:]))
    width = max((len(r[3]) for r in rows), default=1)
    vids = np.zeros((len(rows), width), dtype=np.int64)
    used = np.zeros((len(rows), width), dtype=bool)
    for j, r in enumerate(rows):
        vids[j, :len(r[3])], used[j, :len(r[3])] = r[3], True
    loc = np.array([r[4] for r in rows], dtype=np.float64).reshape(-1, 3)
    return {"gyx": np.array([(r[0], r[1]) for r in rows], dtype=np.int32).reshape(-1, 2),
            "mesh": np.array([r[2] for r in rows], dtype=np.int32), "vids": vids, "used": used, "loc": loc,
            "depth": _project(cf, loc)[2], "grid": (gh, gw)}


def _weights(verts: list[np.ndarray], hits: dict) -> np.ndarray:
    """Inverse-distance weights of each hit over its polygon's vertices (0 for padding)."""
    w = np.zeros(hits["vids"].shape)
    for k, v in enumerate(verts):
        sel = hits["mesh"] == k
        d = np.linalg.norm(v[hits["vids"][sel]] - hits["loc"][sel][:, None, :], axis=2)
        w[sel] = np.where(hits["used"][sel], 1.0 / (d + 1e-6), 0.0)
    return w / w.sum(axis=1, keepdims=True)


def _blend(verts: list[np.ndarray], hits: dict, w: np.ndarray, vids: np.ndarray) -> np.ndarray:
    pts = np.zeros((len(hits["mesh"]), 3))
    for k, v in enumerate(verts):
        sel = hits["mesh"] == k
        pts[sel] = np.einsum("nw,nwc->nc", w[sel], v[vids[sel]])
    return pts


def _visible(rest: dict, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """True where (x, y, depth) lies on the rest frame's front surface (3×3 cell tolerance)."""
    buf = rest["depth_grid"]
    gh, gw = buf.shape
    gx = np.clip(((x - 0.5) / STEP).round().astype(int), 0, gw - 1)
    gy = np.clip(((y - 0.5) / STEP).round().astype(int), 0, gh - 1)
    ok = np.zeros(len(x), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            b = buf[np.clip(gy + dy, 0, gh - 1), np.clip(gx + dx, 0, gw - 1)]
            ok |= np.abs(z - b) < VIS_EPS
    return ok


def _rest(scene, meshes, polys, dg) -> dict:
    hits = _cast(scene, meshes, polys, dg)
    verts = [_world_verts(m, dg) for m in meshes]
    w = _weights(verts, hits)
    buf = np.full(hits["grid"], np.inf)
    buf[hits["gyx"][:, 0], hits["gyx"][:, 1]] = hits["depth"]
    offset = hits["loc"] - _blend(verts, hits, w, hits["vids"])
    return {"hits": hits, "verts": verts, "w": w, "offset": offset, "depth_grid": buf}


def _forward(scene, meshes, rest: dict, dg) -> np.ndarray:
    """Rest samples → where they are now: (n, 3) x, y, depth (L-076 forward map)."""
    verts = [_world_verts(m, dg) for m in meshes]
    pts = _blend(verts, rest["hits"], rest["w"], rest["hits"]["vids"]) + rest["offset"]
    x, y, z = _project(_camera_frame(scene.camera), pts)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _backward(scene, meshes, polys, rest: dict, mirror: list[np.ndarray], dg) -> dict:
    """Every body pixel now → where to copy it from in the rest frame, how: 1 direct, 2 mirror, 3 hidden."""
    hits = _cast(scene, meshes, polys, dg)
    verts = [_world_verts(m, dg) for m in meshes]
    w = _weights(verts, hits)
    cf = _camera_frame(scene.camera)
    direct = _blend(rest["verts"], hits, w, hits["vids"])
    twin_ids = np.zeros_like(hits["vids"])
    for k, mm in enumerate(mirror):
        sel = hits["mesh"] == k
        twin_ids[sel] = mm[hits["vids"][sel]]
    twin = _blend(rest["verts"], hits, w, twin_ids)
    dx, dy, dz = _project(cf, direct)
    tx, ty, tz = _project(cf, twin)
    vis_d, vis_t = _visible(rest, dx, dy, dz), _visible(rest, tx, ty, tz)
    mode = np.where(vis_d, 1, np.where(vis_t, 2, 3)).astype(np.uint8)
    src = np.where(vis_d[:, None], np.stack([dx, dy], 1), np.stack([tx, ty], 1)).astype(np.float32)
    return {"gyx": hits["gyx"].astype(np.int16), "src": src, "mode": mode}


def render_rig() -> dict:
    """Humanoid guide frames (as in the paint spike, body inflated) + per direction the forward and backward
    pixel maps for every frame.

    Returns {"frames": {key: png}, "rig": {d: {"gyx", "grid", "maps": {key: (n, 3)}, "back": {key: {...}}}},
    "checks", "blender"}.
    """
    import bpy
    from bpy_extras.object_utils import world_to_camera_view

    checks: dict = {}
    t0 = time.perf_counter()
    scene = _reset_scene()
    rig: dict = {}
    state: dict = {}

    def on_setup(meshes) -> None:
        for o in bpy.data.objects:  # hidden helper meshes (hide_render only) would still block rays
            if o.type == "MESH" and o not in meshes:
                o.hide_viewport = True
        _inflate(meshes)
        bpy.context.view_layer.update()
        state["mirror"] = _mirror_map(meshes)

    def on_frame(key: str, meshes) -> None:
        dg = bpy.context.evaluated_depsgraph_get()
        if "polys" not in state:
            state["polys"] = _polys(meshes, dg)
        d, tag, i = key.split("_")
        if tag == "idle" and i == "0":  # rest frame = idle frame 0, the one that gets painted
            ts = time.perf_counter()
            state[d] = _rest(scene, meshes, state["polys"], dg)
            h = state[d]["hits"]
            rig[d] = {"gyx": h["gyx"], "grid": h["grid"], "maps": {}, "back": {}}
            checks[f"rays_{d}"] = {"hits": int(len(h["mesh"])), "seconds": round(time.perf_counter() - ts, 1)}
            m0 = meshes[0]  # our projection must agree with Blender's for the same point
            p = m0.matrix_world @ m0.data.vertices[0].co
            ref = world_to_camera_view(scene, scene.camera, p)
            x, y, _ = _project(_camera_frame(scene.camera), np.array([list(p)]))
            checks[f"projection_error_px_{d}"] = round(float(max(abs(x[0] - ref.x * CELL_W),
                                                                 abs(y[0] - (1 - ref.y) * CELL_H))), 3)
        rig[d]["maps"][key] = _forward(scene, meshes, state[d], dg)
        back = _backward(scene, meshes, state["polys"], state[d], state["mirror"], dg)
        rig[d]["back"][key] = back
        n = max(1, len(back["mode"]))
        checks.setdefault(f"back_{d}", {})[key] = {m: round(float((back["mode"] == v).sum() / n), 3)
                                                   for m, v in (("direct", 1), ("mirror", 2), ("hidden", 3))}

    frames = _render_humanoid(scene, checks, on_frame=on_frame, on_setup=on_setup)
    for d in rig:
        start = rig[d]["gyx"][:, ::-1] * STEP + 0.5  # the rest frame must map onto itself, both ways
        checks[f"rest_self_error_px_{d}"] = round(float(np.abs(rig[d]["maps"][f"{d}_idle_0"][:, :2] - start)
                                                         .mean()), 3)
        b = rig[d]["back"][f"{d}_idle_0"]
        checks[f"rest_back_error_px_{d}"] = round(float(np.abs(b["src"] - (b["gyx"][:, ::-1] * STEP + 0.5))
                                                         .mean()), 3)
    checks["seconds"] = round(time.perf_counter() - t0, 1)
    return {"frames": frames, "rig": rig, "checks": checks, "blender": bpy.app.version_string}
