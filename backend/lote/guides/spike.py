"""F-01 slice 1 spike: render guide frames for one humanoid and one hand-written quadruped.

Runs inside the Blender image (`modal_app.py::render_spike_guides`). Answers the bpy questions the spike owns
(spec.md > Wiring Blueprint 1): Skin modifier + `skin_armature_create` headless in bpy 5.2.3, UAL1 clip names,
and what a gray clay guide looks like on the key color. Slice 2 replaces this with `guides/*` proper (body
plans, IK gaits); the camera and cell layout here match spec.md > Guide function so the paint test is
representative.

Sheet assembly (`assemble`) runs on the local side with Pillow; everything else needs `bpy`.
"""

import math
import time
from pathlib import Path

# 4 columns × 3 rows of 400×533 cells fill the 1600² sheet. Spike round 1: Muse Image ignored a 4×4 grid
# with an empty bottom row and spread the 12 figures over three full-height rows anyway (BUILD_LOG L-072).
SHEET = 1600
COLS, ROWS = 4, 3
CELL_W, CELL_H = SHEET // COLS, SHEET // ROWS
FOOT = (CELL_W // 2, int(CELL_H * 0.85))  # foot point in every cell: camera shift puts the origin here
PITCH_DEG = 30  # camera elevation above the ground: fixed ¾ view
DIRS = {"s": 0.0, "e": math.pi / 2, "n": math.pi}  # body yaw; the camera never moves
CLIPS = {"idle": 4, "walk": 8}  # cells 0–3 idle, 4–11 walk
UAL_CLIPS = {"idle": "Idle_Loop", "walk": "Walk_Loop"}
ASSETS = Path("/assets/quaternius")

GRAY, WARM, COOL = (0.55, 0.55, 0.55), (0.66, 0.53, 0.44), (0.44, 0.53, 0.66)  # left limbs warm, right cool


# ---------------------------------------------------------------- scene


def _reset_scene():
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = CELL_W, CELL_H
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.display.render_aa = "8"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_object_outline = True
    sh.object_outline_color = (0.05, 0.05, 0.05)
    return scene


def _materials():
    import bpy

    mats = []
    for name, rgb in (("gray", GRAY), ("warm", WARM), ("cool", COOL)):
        m = bpy.data.materials.new(name)
        m.diffuse_color = (*rgb, 1.0)
        mats.append(m)
    return mats


def _camera(scene, ortho_scale: float):
    import bpy

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.sensor_fit = "VERTICAL"  # ortho_scale = visible height in metres
    cam_data.ortho_scale = ortho_scale
    cam_data.shift_y = FOOT[1] / CELL_H - 0.5  # origin lands on the foot row
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    pitch = math.radians(PITCH_DEG)
    cam.location = (0.0, -20 * math.cos(pitch), 20 * math.sin(pitch))
    cam.rotation_euler = (math.pi / 2 - pitch, 0.0, 0.0)
    scene.camera = cam


def _world_bounds(objs) -> tuple[float, float, float]:
    """(height, length along Y, width along X) of the evaluated meshes in world space."""
    import bpy
    from mathutils import Vector

    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objs:
        ev = o.evaluated_get(dg)
        pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    xs, ys, zs = ([p[i] for p in pts] for i in range(3))
    return max(zs) - min(0.0, min(zs)), max(ys) - min(ys), max(xs) - min(xs)


def _render(scene) -> bytes:
    path = "/tmp/frame.png"
    scene.render.filepath = path
    import bpy

    bpy.ops.render.render(write_still=True)
    return Path(path).read_bytes()


def _tint_by_side(obj, side_of_group) -> None:
    """Material per face from the side (+1 left, -1 right, 0 center) of each vertex's strongest group."""
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    vside = []
    for v in me.vertices:
        best = max(v.groups, key=lambda g: g.weight, default=None)
        vside.append(side_of_group(names[best.group]) if best else 0)
    for p in me.polygons:
        s = sum(vside[i] for i in p.vertices)
        p.material_index = 1 if s > 0 else 2 if s < 0 else 0


# ---------------------------------------------------------------- humanoid


def _import(path: Path) -> list:
    import bpy

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def _humanoid(checks: dict):
    """Superhero (male) with UAL1 Idle_Loop / Walk_Loop. Returns (top object, armature, meshes, actions)."""
    import bpy

    # UAL1 first: keep its actions, drop its objects.
    ual_objs = _import(ASSETS / "animations" / "UAL1_Standard.glb")
    checks["ual1_clips"] = sorted(a.name for a in bpy.data.actions)
    actions = {}
    for tag, clip in UAL_CLIPS.items():
        hits = sorted((a for a in bpy.data.actions if a.name.startswith(clip)), key=lambda a: len(a.name))
        if not hits:
            raise RuntimeError(f"UAL1 clip {clip} not found; have {checks['ual1_clips']}")
        hits[0].use_fake_user = True
        actions[tag] = hits[0]
    checks["ual1_used"] = {t: a.name for t, a in actions.items()}
    checks["ual1_frame_ranges"] = {t: list(a.frame_range) for t, a in actions.items()}
    for o in ual_objs:
        bpy.data.objects.remove(o, do_unlink=True)

    objs = _import(ASSETS / "characters" / "Superhero_Male_FullBody.gltf")
    arm = next(o for o in objs if o.type == "ARMATURE")
    meshes = [o for o in objs if o.type == "MESH"]
    top = arm
    while top.parent:
        top = top.parent
    top.rotation_mode = "XYZ"  # the glTF importer sets quaternions, which would ignore rotation_euler (yaw)
    # Only meshes the rig deforms are the body; others (an `Icosphere` helper) are hidden.
    skinned = [m for m in meshes if any(md.type == "ARMATURE" for md in m.modifiers)] or meshes
    for m in meshes:
        if m not in skinned:
            m.hide_render = True
    checks["humanoid_hidden"] = [m.name for m in meshes if m not in skinned]
    meshes = skinned
    mats = _materials()
    for m in meshes:
        m.data.materials.clear()
        for mat in mats:
            m.data.materials.append(mat)
        if m.vertex_groups:
            _tint_by_side(m, lambda n: 1 if n.endswith("_l") else -1 if n.endswith("_r") else 0)
    checks["humanoid_meshes"] = [m.name for m in meshes]
    return top, arm, meshes, actions


def _set_action(arm, action) -> None:
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = action
    if getattr(ad, "action_slot", "n/a") is None and getattr(action, "slots", None):
        ad.action_slot = action.slots[0]  # Blender 4.4+ slotted actions


def _render_humanoid(scene, checks: dict, on_frame=None, on_setup=None) -> dict:
    """`on_setup(meshes)` runs once before any render; `on_frame(key, meshes)` after each render with the pose
    still set (the rig spike uses both)."""
    import bpy

    top, arm, meshes, actions = _humanoid(checks)
    if on_setup:
        on_setup(meshes)
    _set_action(arm, actions["idle"])
    scene.frame_set(int(actions["idle"].frame_range[0]))
    height, length, width = _world_bounds(meshes)
    checks["humanoid_bounds_m"] = [round(height, 3), round(length, 3), round(width, 3)]
    _camera(scene, ortho_scale=height * 1.3)
    pelvis = arm.pose.bones["pelvis"]
    vl = bpy.context.view_layer

    frames = {}
    for d, yaw in DIRS.items():
        top.rotation_euler = (top.rotation_euler[0], top.rotation_euler[1], yaw)
        for tag, n in CLIPS.items():
            act = actions[tag]
            _set_action(arm, act)
            start, end = act.frame_range
            for i in range(n):
                f = start + (end - start) * i / n
                scene.frame_set(int(f), subframe=f - int(f))
                # In place: pin the pelvis over the origin (keeps the vertical bob).
                top.location = (0.0, 0.0, 0.0)
                vl.update()
                p = arm.matrix_world @ pelvis.head
                top.location = (-p.x, -p.y, 0.0)
                vl.update()
                frames[f"{d}_{tag}_{i}"] = _render(scene)
                if on_frame:
                    on_frame(f"{d}_{tag}_{i}", meshes)
    return frames


# ---------------------------------------------------------------- quadruped (hand-written body plan)

# Body along Y, head toward -Y (faces the camera at yaw 0). Units: shoulder height ≈ 0.7.
QUAD_VERTS = {
    "mid": (0, 0.0, 0.66, 0.20),
    "hip": (0, 0.40, 0.64, 0.19),
    "shoulder": (0, -0.40, 0.68, 0.20),
    # Raised neck: round 2's low dog-like head fought the upright dragon reference (L-073).
    "neck": (0, -0.56, 1.08, 0.12),
    "head": (0, -0.66, 1.36, 0.14),
    "snout": (0, -0.88, 1.32, 0.07),
    "tail1": (0, 0.62, 0.66, 0.07),
    "tail2": (0, 0.86, 0.60, 0.05),
    "tail3": (0, 1.06, 0.52, 0.03),
}
for _side, _x in (("l", 0.17), ("r", -0.17)):
    for _end, _y in (("f", -0.40), ("h", 0.40)):
        QUAD_VERTS[f"{_end}hip_{_side}"] = (_x, _y, 0.60, 0.09)
        QUAD_VERTS[f"{_end}knee_{_side}"] = (_x, _y + (0.03 if _end == "f" else -0.05), 0.32, 0.06)
        QUAD_VERTS[f"{_end}foot_{_side}"] = (_x, _y, 0.04, 0.05)

# Edges root-outward (root = mid). Labels name the bones the gait drives.
QUAD_EDGES = [
    ("mid", "hip", "spine_h"), ("mid", "shoulder", "spine_f"), ("shoulder", "neck", "neck"),
    ("neck", "head", "head"), ("head", "snout", "snout"),
    ("hip", "tail1", "tail1"), ("tail1", "tail2", "tail2"), ("tail2", "tail3", "tail3"),
]
for _side in ("l", "r"):
    for _end, _root in (("f", "shoulder"), ("h", "hip")):
        QUAD_EDGES += [
            (_root, f"{_end}hip_{_side}", f"{_end}girdle_{_side}"),
            (f"{_end}hip_{_side}", f"{_end}knee_{_side}", f"{_end}thigh_{_side}"),
            (f"{_end}knee_{_side}", f"{_end}foot_{_side}", f"{_end}shin_{_side}"),
        ]

# Lateral-sequence walk: LH, LF, RH, RF a quarter cycle apart (spec.md > Guide function, 4 legs).
WALK_PHASE = {"h_l": 0.0, "f_l": 0.25, "h_r": 0.5, "f_r": 0.75}


def _skin_mesh():
    import bpy

    names = list(QUAD_VERTS)
    idx = {n: i for i, n in enumerate(names)}
    me = bpy.data.meshes.new("quad")
    me.from_pydata([QUAD_VERTS[n][:3] for n in names], [(idx[a], idx[b]) for a, b, _ in QUAD_EDGES], [])
    obj = bpy.data.objects.new("quad", me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    # The operator (not modifiers.new) also creates the skin vertex layer.
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.object.modifier_add(type="SKIN")
    mod = obj.modifiers[-1]
    mod.name = "Skin"
    mod.use_smooth_shade = True
    sv = me.skin_vertices[0].data
    for n, i in idx.items():
        r = QUAD_VERTS[n][3]
        sv[i].radius = (r, r)
        sv[i].use_root = n == "mid"
    obj.modifiers.new("Subdiv", "SUBSURF").levels = 1
    return obj, idx


def _bone_map(arm, idx: dict) -> dict[str, str]:
    """Match each labelled edge to the armature bone whose head/tail sit on its vertices."""
    from mathutils import Vector

    out = {}
    for a, b, label in QUAD_EDGES:
        va, vb = Vector(QUAD_VERTS[a][:3]), Vector(QUAD_VERTS[b][:3])
        best, err = None, 1e9
        for bone in arm.data.bones:
            e = min((bone.head_local - va).length + (bone.tail_local - vb).length,
                    (bone.head_local - vb).length + (bone.tail_local - va).length)
            if e < err:
                best, err = bone.name, e
        if err < 0.02:
            out[label] = best
    return out


def _manual_armature(obj, idx: dict):
    """Fallback if `skin_armature_create` fails headless: one bone per edge, one group per child vertex."""
    import bpy

    arm_data = bpy.data.armatures.new("quad_arm")
    arm = bpy.data.objects.new("quad_arm", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ending_at = {}
    for a, b, label in QUAD_EDGES:
        eb = arm_data.edit_bones.new(label)
        eb.head, eb.tail = QUAD_VERTS[a][:3], QUAD_VERTS[b][:3]
        if a in ending_at:
            eb.parent = arm_data.edit_bones[ending_at[a]]
            eb.use_connect = True
        ending_at[b] = label
    bpy.ops.object.mode_set(mode="OBJECT")
    for _, b, label in QUAD_EDGES:
        obj.vertex_groups.new(name=label).add([idx[b]], 1.0, "REPLACE")
    obj.vertex_groups["spine_f"].add([idx["mid"]], 1.0, "REPLACE")
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    return arm


def _quadruped(checks: dict):
    import bpy

    obj, idx = _skin_mesh()
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    obj.select_set(True)
    vl.objects.active = obj
    before = set(bpy.data.objects)
    try:
        with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj]):
            bpy.ops.object.skin_armature_create(modifier="Skin")
        arm = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
        checks["skin_armature_create"] = {"ok": True, "bones": len(arm.data.bones)}
    except Exception as e:  # noqa: BLE001 - the spike reports it and falls back
        checks["skin_armature_create"] = {"ok": False, "error": repr(e)}
        arm = _manual_armature(obj, idx)
    bones = _bone_map(arm, idx)
    checks["skin_bone_map"] = {"matched": len(bones), "of": len(QUAD_EDGES)}
    if len(bones) < len(QUAD_EDGES):
        raise RuntimeError(f"bone map incomplete: {sorted(set(e[2] for e in QUAD_EDGES) - set(bones))}")

    # Make the skin real geometry so faces can carry left/right materials; weights carry over.
    vl.objects.active = obj
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.object.modifier_apply(modifier="Skin")
    checks["skin_applied_faces"] = len(obj.data.polygons)
    mats = _materials()
    for mat in mats:
        obj.data.materials.append(mat)
    side = {}
    for label, bone in bones.items():
        side[bone] = 1 if label.endswith("_l") else -1 if label.endswith("_r") else 0
    _tint_by_side(obj, lambda n: side.get(n, 0))
    if obj.parent is None:
        obj.parent = arm
    return obj, arm, bones


def _rot_x(arm, bone: str, angle: float, axis: str = "X"):
    """Rotate a pose bone about a world axis (armature at rest yaw), expressed in the bone's rest frame."""
    from mathutils import Matrix

    m = arm.data.bones[bone].matrix_local.to_3x3()
    pb = arm.pose.bones[bone]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (m.inverted() @ Matrix.Rotation(angle, 3, axis) @ m).to_quaternion()


def _quad_pose(arm, bones: dict, tag: str, t: float) -> float:
    """Pose for cycle time t in [0, 1). Returns the body's vertical bob."""
    tau = 2 * math.pi
    if tag == "walk":
        for leg, off in WALK_PHASE.items():
            end, side = leg.split("_")
            p = tau * (t + off)
            _rot_x(arm, bones[f"{end}thigh_{side}"], 0.6 * math.sin(p))
            _rot_x(arm, bones[f"{end}shin_{side}"], -0.7 * max(0.0, math.cos(p)) if end == "f" else
                   0.7 * max(0.0, math.cos(p)))
        for i, k in enumerate(("tail1", "tail2", "tail3")):
            _rot_x(arm, bones[k], 0.18 * math.sin(tau * t - i * 0.6), axis="Z")
        _rot_x(arm, bones["head"], 0.06 * math.sin(2 * tau * t))
        return 0.02 * math.cos(2 * tau * t)
    for leg in WALK_PHASE:
        end, side = leg.split("_")
        _rot_x(arm, bones[f"{end}thigh_{side}"], 0.0)
        _rot_x(arm, bones[f"{end}shin_{side}"], 0.0)
    for i, k in enumerate(("tail1", "tail2", "tail3")):
        _rot_x(arm, bones[k], 0.12 * math.sin(tau * t - i * 0.5), axis="Z")
    _rot_x(arm, bones["head"], 0.05 * math.sin(tau * t))
    _rot_x(arm, bones["neck"], 0.03 * math.sin(tau * t))
    return 0.008 * math.sin(tau * t)


def _render_quadruped(scene, checks: dict) -> dict:
    import bpy

    obj, arm, bones = _quadruped(checks)
    vl = bpy.context.view_layer
    vl.update()
    height, length, width = _world_bounds([obj])
    checks["quadruped_bounds"] = [round(height, 3), round(length, 3), round(width, 3)]
    _camera(scene, ortho_scale=max(height * 1.3, length * 1.5 * CELL_H / CELL_W))
    frames = {}
    for d, yaw in DIRS.items():
        arm.rotation_euler = (0.0, 0.0, yaw)
        for tag, n in CLIPS.items():
            for i in range(n):
                bob = _quad_pose(arm, bones, tag, i / n)
                arm.location = (0.0, 0.0, bob)
                vl.update()
                frames[f"{d}_{tag}_{i}"] = _render(scene)
    return frames


# ---------------------------------------------------------------- entry points


def render_all() -> dict:
    """Both bodies → {"humanoid": {frames, checks}, "quadruped": {...}, "blender": version}."""
    import bpy

    out = {"blender": bpy.app.version_string}
    for body, fn in (("humanoid", _render_humanoid), ("quadruped", _render_quadruped)):
        checks: dict = {}
        t0 = time.perf_counter()
        scene = _reset_scene()
        try:
            frames = fn(scene, checks)
            checks["ok"] = True
        except Exception as e:  # noqa: BLE001 - one body failing must not hide the other's result
            frames, checks["ok"], checks["error"] = {}, False, repr(e)
        checks["seconds"] = round(time.perf_counter() - t0, 1)
        out[body] = {"frames": frames, "checks": checks}
    return out


def cell_box(index: int) -> tuple[int, int, int, int]:
    r, c = divmod(index, COLS)
    return (c * CELL_W, r * CELL_H, (c + 1) * CELL_W, (r + 1) * CELL_H)


def assemble(frames: dict[str, bytes], key_rgb: tuple[int, int, int]) -> dict:
    """Local side: frames → per direction a key-color sheet + alpha sheet; turnaround; anchors.

    Returns {"sheets": {d: png}, "masks": {d: png}, "turnaround": png, "anchors": {...}}.
    """
    import io

    from PIL import Image

    def keyed(rgba: Image.Image) -> Image.Image:
        bg = Image.new("RGBA", rgba.size, (*key_rgb, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")

    def png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    sheets, masks, anchors = {}, {}, {"cell": [CELL_W, CELL_H], "cols": COLS, "rows": ROWS,
                                  "foot": FOOT, "frames": []}
    for d in DIRS:
        sheet = Image.new("RGB", (SHEET, SHEET), key_rgb)
        alpha = Image.new("L", sheet.size, 0)
        cell = 0
        for tag, n in CLIPS.items():
            for i in range(n):
                rgba = Image.open(io.BytesIO(frames[f"{d}_{tag}_{i}"])).convert("RGBA")
                box = cell_box(cell)
                sheet.paste(keyed(rgba), box[:2])
                alpha.paste(rgba.getchannel("A"), box[:2])
                anchors["frames"].append({"dir": d, "tag": tag, "i": i, "cell": cell, "box": box,
                                          "foot": (box[0] + FOOT[0], box[1] + FOOT[1])})
                cell += 1
        sheets[d], masks[d] = png(sheet), png(alpha)
    turn = Image.new("RGB", (CELL_W * len(DIRS), CELL_H), key_rgb)
    for k, d in enumerate(DIRS):
        turn.paste(keyed(Image.open(io.BytesIO(frames[f"{d}_idle_0"])).convert("RGBA")), (k * CELL_W, 0))
    return {"sheets": sheets, "masks": masks, "turnaround": png(turn), "anchors": anchors}
