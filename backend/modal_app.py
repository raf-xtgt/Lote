"""Modal app: images, functions, volume, dicts, secret (ARCHITECTURE.md > Cloud and Infrastructure).

Setup 01 skeleton: web endpoint with /api/health, plus smoke functions that prove the Blender and
Chromium images.
  uv run modal run modal_app.py::smoke     build all images, run the smoke checks
  uv run modal run modal_app.py::guide_spike   F-01 slice 1: guide sheets for the painting spike
  uv run modal run modal_app.py::rig_spike     F-01 slice 1: guide frames + pixel maps for the 2D rig test
  uv run modal serve modal_app.py          dev URL with hot reload
  uv run modal deploy modal_app.py         live URL
"""

import tomllib
from pathlib import Path

import modal

HERE = Path(__file__).parent
PY = "3.13"  # L-024
UI_DIST = HERE.parent / "frontend" / "dist"

# Runtime deps come from pyproject.toml so the image and the local env never drift. Images are defined only
# on the local side; inside a container this module is re-imported without pyproject.toml, so skip the read.
APP_DEPS: list[str] = []
if modal.is_local():
    APP_DEPS = tomllib.loads((HERE / "pyproject.toml").read_text())["project"]["dependencies"]

base_image = modal.Image.debian_slim(python_version=PY).uv_pip_install(*APP_DEPS)
web_image = base_image.add_local_python_source("lote")
if UI_DIST.is_dir():
    web_image = web_image.add_local_dir(UI_DIST, "/assets/ui")

blender_image = (
    modal.Image.debian_slim(python_version=PY)
    .apt_install("xorg", "libxkbcommon0")  # bpy runtime libs (Modal Blender example)
    .uv_pip_install("bpy==5.2.3")
    .add_local_dir(HERE / "assets" / "quaternius", "/assets/quaternius", ignore=["_download", "_download/**"])
    .add_local_python_source("lote")
)

playtest_image = (
    modal.Image.debian_slim(python_version=PY)
    .uv_pip_install("playwright==1.64.1")
    .run_commands("playwright install --with-deps chromium")
)

app = modal.App("lote")
secret = modal.Secret.from_name("meta-model-api")
runs = modal.Volume.from_name("lote-runs")
progress = modal.Dict.from_name("lote-progress")
spend = modal.Dict.from_name("lote-spend")


@app.function(image=web_image, secrets=[secret], volumes={"/runs": runs})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from lote.api.app import create_app

    return create_app(volume=runs, progress=progress)


@app.function(image=base_image, secrets=[secret], volumes={"/runs": runs})
def storage_smoke() -> dict:
    """Secret visible, Volume write + commit + read back, both Dicts reachable."""
    import os
    import time

    stamp = str(time.time())
    path = Path("/runs/_smoke/stamp.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stamp)
    runs.commit()
    runs.reload()
    progress.put("smoke", stamp)
    spend.put("smoke", 0.0)
    return {
        "secret": bool(os.environ.get("MODEL_API_KEY")),
        "volume": path.read_text() == stamp,
        "dict_progress": progress.get("smoke") == stamp,
        "dict_spend": spend.get("smoke") == 0.0,
    }


@app.function(image=blender_image, cpu=2, memory=4096, timeout=600)
def guide_smoke() -> dict:
    """Render a cube with an orthographic camera. Tries Workbench, then Cycles CPU; reports which worked."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add(size=2)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 5
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (6, -6, 5)
    cam.rotation_euler = (1.0, 0.0, 0.785)
    scene.camera = cam
    light = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    scene.collection.objects.link(light)
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    errors = {}
    for engine in ("BLENDER_WORKBENCH", "CYCLES"):
        try:
            scene.render.engine = engine
            if engine == "CYCLES":
                scene.cycles.device = "CPU"
                scene.cycles.samples = 4
            out = f"/tmp/cube_{engine.lower()}.png"
            scene.render.filepath = out
            bpy.ops.render.render(write_still=True)
            return {"engine": engine, "png": Path(out).read_bytes(), "blender": bpy.app.version_string,
                    "errors": errors}
        except Exception as e:  # noqa: BLE001 - try the next engine
            errors[engine] = repr(e)
    return {"engine": None, "png": b"", "blender": bpy.app.version_string, "errors": errors}


@app.function(image=playtest_image, cpu=2, memory=4096, timeout=300)
async def playtest_smoke() -> dict:
    """Headless Chromium: load a page, press a key, read it back through an exposed function, screenshot."""
    from playwright.async_api import async_playwright

    seen: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.expose_function("qaLog", lambda line: seen.append(line))
        await page.set_content(
            "<body style='background:#283;color:#fff;font:40px sans-serif'><p id=t>Lote playtest smoke</p>"
            "<script>addEventListener('keydown', e => qaLog('key ' + e.key))</script></body>"
        )
        await page.keyboard.press("ArrowRight")
        await page.wait_for_timeout(200)
        png = await page.screenshot(type="jpeg", quality=80)
        version = browser.version
        await browser.close()
    return {"chromium": version, "jpeg": png, "qa_lines": seen}


@app.function(image=blender_image, cpu=2, memory=4096, timeout=900)
def render_spike_guides() -> dict:
    """F-01 spike: humanoid + hand-written quadruped guide frames, S/E/N, idle 4 + walk 8."""
    from lote.guides.spike import render_all

    return render_all()


@app.local_entrypoint()
def guide_spike():
    """Writes runs/spike/guides/<body>/: sheet_{s,e,n}.png, mask_{s,e,n}.png, turnaround.png, anchors.json."""
    import json

    from lote.guides.spike import assemble

    key = (255, 0, 255)  # spike key color: magenta (a run picks its own later, spec.md > Understand stage)
    out = HERE / "runs" / "spike" / "guides"
    out.mkdir(parents=True, exist_ok=True)
    result = render_spike_guides.remote()
    report = {"blender": result["blender"], "key_rgb": key}
    for body in ("humanoid", "quadruped"):
        frames, checks = result[body]["frames"], result[body]["checks"]
        report[body] = checks
        if not checks["ok"]:
            print(f"[FAIL] {body}: {checks.get('error')}")
            continue
        sheet = assemble(frames, key)
        d = out / body
        d.mkdir(parents=True, exist_ok=True)
        for k, png in sheet["sheets"].items():
            (d / f"sheet_{k}.png").write_bytes(png)
        for k, png in sheet["masks"].items():
            (d / f"mask_{k}.png").write_bytes(png)
        (d / "turnaround.png").write_bytes(sheet["turnaround"])
        (d / "anchors.json").write_text(json.dumps(sheet["anchors"], indent=1))
        print(f"[OK] {body}: {len(frames)} frames in {checks['seconds']}s -> {d}")
    (out / "checks.json").write_text(json.dumps(report, indent=2, default=str))
    h, q = report["humanoid"], report["quadruped"]
    skin = q.get("skin_armature_create", {})
    print(f"[{'OK' if h.get('ual1_used') else 'FAIL'}] UAL1 clips used: {h.get('ual1_used')}")
    print(f"[{'OK' if skin.get('ok') else 'FAIL'}] skin_armature_create: {skin}")
    print(f"[{'OK' if q.get('skin_applied_faces') else 'FAIL'}] Skin modifier applied: "
          f"{q.get('skin_applied_faces')} faces; bone map {q.get('skin_bone_map')}")
    print(f"Blender {result['blender']}. Checks in {out / 'checks.json'}")


@app.function(image=blender_image, cpu=2, memory=4096, timeout=900)
def render_rig_guides() -> dict:
    """F-01 spike (L-076): humanoid guide frames + rest → frame pixel maps for the 2D rig test."""
    from lote.guides.rig_spike import render_rig

    return render_rig()


@app.local_entrypoint()
def rig_spike():
    """Writes runs/spike/rig/guides/: sheet_/mask_{s,e,n}.png, anchors.json, rig.npz, checks.json."""
    import json

    import numpy as np

    from lote.guides.spike import assemble

    out = HERE / "runs" / "spike" / "rig" / "guides"
    out.mkdir(parents=True, exist_ok=True)
    result = render_rig_guides.remote()
    sheet = assemble(result["frames"], (255, 0, 255))
    for k, png in sheet["sheets"].items():
        (out / f"sheet_{k}.png").write_bytes(png)
    for k, png in sheet["masks"].items():
        (out / f"mask_{k}.png").write_bytes(png)
    (out / "turnaround.png").write_bytes(sheet["turnaround"])
    (out / "anchors.json").write_text(json.dumps(sheet["anchors"], indent=1))
    arrays = {}
    for d, r in result["rig"].items():
        arrays[f"gyx_{d}"] = r["gyx"]
        arrays[f"grid_{d}"] = np.array(r["grid"])
        for key, m in r["maps"].items():
            arrays[f"map_{key}"] = m
        for key, b in r["back"].items():
            for part, v in b.items():
                arrays[f"back_{part}_{key}"] = v
    np.savez_compressed(out / "rig.npz", **arrays)
    checks = result["checks"]
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str))
    for d in result["rig"]:
        rays, proj = checks[f"rays_{d}"], checks[f"projection_error_px_{d}"]
        self_err, back_err = checks[f"rest_self_error_px_{d}"], checks[f"rest_back_error_px_{d}"]
        hidden = max(v["hidden"] for v in checks[f"back_{d}"].values())
        mirror = max(v["mirror"] for v in checks[f"back_{d}"].values())
        ok = rays["hits"] > 1000 and proj < 1 and self_err < 1 and back_err < 2
        print(f"[{'OK' if ok else 'FAIL'}] {d}: {rays['hits']} surface samples in {rays['seconds']}s, "
              f"projection error {proj} px, rest maps onto itself within {self_err} / {back_err} px; "
              f"worst frame: mirror {mirror:.0%}, hidden {hidden:.0%}")
    print(f"Blender {result['blender']}, {checks['seconds']}s. Files in {out}")


@app.local_entrypoint()
def smoke():
    out = HERE / "runs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    storage = storage_smoke.remote()
    print(f"[{'OK' if all(storage.values()) else 'FAIL'}] storage: {storage}")

    guide = guide_smoke.remote()
    if guide["png"]:
        (out / "guide_cube.png").write_bytes(guide["png"])
    print(f"[{'OK' if guide['png'] else 'FAIL'}] guide: Blender {guide['blender']}, "
          f"engine {guide['engine']}, {len(guide['png'])} bytes, errors {guide['errors']}")

    play = playtest_smoke.remote()
    (out / "playtest.jpg").write_bytes(play["jpeg"])
    ok = bool(play["jpeg"]) and play["qa_lines"] == ["key ArrowRight"]
    print(f"[{'OK' if ok else 'FAIL'}] playtest: Chromium {play['chromium']}, {len(play['jpeg'])} bytes, "
          f"qa_lines {play['qa_lines']}")
    print(f"Saved images to {out}")
