"""F-01 slice 1 spike, 2D rig test (BUILD_LOG L-076): paint one rest frame per direction, animate it with the
guide.

Rounds 1–3 showed Muse Image keeps the character but won't copy guide poses (L-075). Here it paints only the
rest frame (idle frame 0) for S, E and N: three candidates each, the one whose silhouette best matches the
guide is kept, and Spark checks it. Every other frame is made by moving the painted pixels with the rest →
frame maps from `modal run modal_app.py::rig_spike` (each pixel follows its body surface point; nearer pixels
are drawn last). L-077: the body is copied backwards instead (each frame pixel looks up where it sat in the
rest painting, or where its left/right twin sat), the forward warp only adds clothing just outside the body,
and the edge is despilled. Pixels neither view showed are holes, filled from neighbours and counted.

Same checks as rounds 2–3 on the 4×3 sheets (Spark pose + facing, same-character), plus hole share. Writes
runs/spike/rig/paint/: rest candidates, `sheet_{d}.png`, 64 px sheets and `walk_{d}_64.gif`.

Run from backend/:  uv run --env-file .env python scripts/f01_rig_spike.py
Needs runs/spike/rig/guides/ first. Cost about $0.15.
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

import f01_paint_spike as ps
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent.parent
GUIDES = HERE / "runs" / "spike" / "rig" / "guides"
OUT = HERE / "runs" / "spike" / "rig" / "paint"
REPORT = HERE / "runs" / "spike" / "rig" / "report.json"
BODY = "humanoid"
CANDIDATES = 3
STEP = 2  # must match lote/guides/rig_spike.py
CELL_W, CELL_H = 400, 533
SPRITE_H = 64  # output height (L-011)
MAX_HOLES = 0.08  # share of the guide figure the warp couldn't cover


def rest_prompt(desc: str, d: str, n_prior: int) -> str:
    prior = ""
    if n_prior == 1:
        prior = "Image 3 is the same character already painted facing another direction; match it. "
    elif n_prior > 1:
        prior = (f"Images 3 to {2 + n_prior} are the same character already painted facing other directions; "
                 f"match them. ")
    return (f"Image 1 is the style reference. Image 2 is the design card of the character: {desc}. "
            f"{prior}The last image is one gray clay figure on flat magenta, {ps.FACING[d]}, in a relaxed "
            f"standing pose. It is a template: paint the character from the design card exactly over it, in "
            f"the style of image 1, so the painted figure lines up with the gray one: the same stance, the "
            f"same leg positions, both arms hanging at the same angles, the same head turn, the same size "
            f"and position. No action pose, nothing in the hands. One figure only. {ps.BG_RULE}")


def norm_iou(p: np.ndarray, g: np.ndarray, side: int = 160) -> float:
    """Silhouette IoU after scaling both to the same height, bottom-centre aligned."""
    def fit(m: np.ndarray) -> np.ndarray | None:
        b = ps.fig_bbox(m)
        if b is None:
            return None
        m = m[b[1]:b[3], b[0]:b[2]]
        w = max(1, min(2 * side, round(m.shape[1] * side / m.shape[0])))
        r = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, side))) > 127
        out = np.zeros((side, 2 * side), dtype=bool)
        x0 = side - w // 2
        out[:, x0:x0 + w] = r
        return out
    a, b = fit(p), fit(g)
    return 0.0 if a is None or b is None else ps.iou(a, b)


async def rest_check(api: ps.Api, guide: bytes, painted: bytes, facing: str) -> dict:
    schema = {"type": "object",
              "properties": {"pose_match": {"type": "boolean"}, "facing_ok": {"type": "boolean"},
                             "notes": {"type": "string"}},
              "required": ["pose_match", "facing_ok", "notes"], "additionalProperties": False}
    ask = (f"Image 1 is a gray clay template figure. Image 2 is the same figure painted as a character. "
           f"pose_match is true only if the painted figure has the template's pose: the same stance, leg "
           f"positions, arm angles and head turn (armor bulk doesn't matter). facing_ok is true if it faces "
           f"{facing} like the template. Be strict; notes says what differs.")
    content = [{"type": "input_text", "text": ask},
               {"type": "input_image", "image_url": ps.data_url(guide)},
               {"type": "input_image", "image_url": ps.data_url(painted)}]
    return await api.spark(content, "RestCheck", schema)


# ---------------------------------------------------------------- warp


def fill_nearest(values: np.ndarray, valid: np.ndarray, limit: np.ndarray | None = None,
                 iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Grow valid cells into invalid neighbours (4-connected), optionally only where `limit` is true."""
    v, ok = values.copy(), valid.copy()
    for _ in range(iters):
        grown = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src_ok = np.roll(ok, (dy, dx), axis=(0, 1))
            take = ~ok & src_ok if limit is None else ~ok & src_ok & limit
            if take.any():
                v[take] = np.roll(v, (dy, dx), axis=(0, 1))[take]
                ok = ok | take
                grown = True
        if not grown:
            break
    return v, ok


def grid_fields(gyx: np.ndarray, grid: tuple[int, int], m: np.ndarray) -> np.ndarray:
    """Per grid cell: (dx, dy, depth) of the sampled surface point, spread to every cell from its nearest
    sample.
    Spreading displacement (not position) lets painted pixels outside the guide silhouette move with the body
    part next to them."""
    gh, gw = grid
    f = np.zeros((gh, gw, 3), dtype=np.float32)
    ok = np.zeros((gh, gw), dtype=bool)
    gy, gx = gyx[:, 0], gyx[:, 1]
    f[gy, gx, 0] = m[:, 0] - (gx * STEP + 0.5)
    f[gy, gx, 1] = m[:, 1] - (gy * STEP + 0.5)
    f[gy, gx, 2] = m[:, 2]
    ok[gy, gx] = True
    f, _ = fill_nearest(f, ok)
    return f


def sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a grid field at pixel coordinates."""
    gh, gw, _ = field.shape
    gx = np.clip((x - 0.5) / STEP, 0, gw - 1.001)
    gy = np.clip((y - 0.5) / STEP, 0, gh - 1.001)
    x0, y0 = gx.astype(int), gy.astype(int)
    fx, fy = (gx - x0)[:, None], (gy - y0)[:, None]
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)


def warp(rgb: np.ndarray, alpha: np.ndarray, field: np.ndarray,
         guide_alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """Move every painted pixel (as 2×2 sub-samples) by the field; far first, near last. Then close cracks
    and fill holes inside the guide's silhouette from neighbours. Returns (RGBA uint8, hole share)."""
    ys, xs = np.nonzero(alpha)
    subs = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    sx = np.concatenate([xs + ox for ox, _ in subs])
    sy = np.concatenate([ys + oy for _, oy in subs])
    col = np.concatenate([rgb[ys, xs]] * 4)
    f = sample(field, sx, sy)
    tx, ty, depth = (sx + f[:, 0]).astype(int), (sy + f[:, 1]).astype(int), f[:, 2]
    inside = (tx >= 0) & (tx < CELL_W) & (ty >= 0) & (ty < CELL_H)
    order = np.argsort(-depth[inside], kind="stable")  # far → near, so nearer pixels overwrite
    tx, ty, col = tx[inside][order], ty[inside][order], col[inside][order]
    out = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    got = np.zeros((CELL_H, CELL_W), dtype=bool)
    out[ty, tx] = col
    got[ty, tx] = True
    # Cracks: empty pixels mostly surrounded by painted ones.
    for _ in range(2):
        n = sum(np.roll(got, (dy, dx), axis=(0, 1)).astype(int)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        crack = ~got & (n >= 5)
        if crack.any():
            filled, _ = fill_nearest(out, got, limit=crack, iters=2)
            out[crack], got = filled[crack], got | crack
    holes = guide_alpha & ~got
    share = float(holes.sum() / max(1, guide_alpha.sum()))
    if holes.any():
        filled, ok = fill_nearest(out, got, limit=holes | got, iters=60)
        out[holes & ok], got = filled[holes & ok], got | (holes & ok)
    return np.dstack([out, got.astype(np.uint8) * 255]), round(share, 4)


def despill(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Key colour off the figure: drop pixels close to the key, pull the magenta tint out of a 3 px rim."""
    a = rgb.astype(int)
    alpha = alpha & (np.abs(a - np.array(ps.KEY)).max(axis=2) >= 110)
    inner = np.asarray(Image.fromarray(alpha.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))) > 127
    rim = alpha & ~inner
    excess = np.where(rim, np.clip(np.minimum(a[..., 0], a[..., 2]) - a[..., 1], 0, None), 0)
    a[..., 0] -= excess
    a[..., 2] -= excess
    return a.clip(0, 255).astype(np.uint8), alpha


MIRROR_SHADE = 0.82  # the far side, copied from its near twin, sits a little in shadow


def backward(rgb: np.ndarray, alpha: np.ndarray, b: dict, grid: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Every body pixel of the frame copies its colour from where it sat in the rest painting (or where its
    left/right twin sat). Returns (rgb, painted mask, body mask, mirror mask)."""
    gh, gw = grid
    srcg = np.zeros((gh, gw, 2), dtype=np.float32)
    mode = np.zeros((gh, gw), dtype=np.uint8)
    gy, gx = b["gyx"][:, 0].astype(int), b["gyx"][:, 1].astype(int)
    srcg[gy, gx], mode[gy, gx] = b["src"], b["mode"]
    ys, xs = np.mgrid[0:CELL_H, 0:CELL_W]
    cy, cx = np.minimum(ys // STEP, gh - 1), np.minimum(xs // STEP, gw - 1)
    m = mode[cy, cx]
    sx = np.clip(np.floor(srcg[cy, cx, 0] + xs - cx * STEP), 0, CELL_W - 1).astype(int)
    sy = np.clip(np.floor(srcg[cy, cx, 1] + ys - cy * STEP), 0, CELL_H - 1).astype(int)
    ok = ((m == 1) | (m == 2)) & alpha[sy, sx]
    col = rgb[sy, sx].astype(np.float32)
    col[m == 2] *= MIRROR_SHADE
    out = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    out[ok] = col[ok].astype(np.uint8)
    return out, ok, m > 0, ok & (m == 2)


def compose(rgb: np.ndarray, alpha: np.ndarray, b: dict, field: np.ndarray,
            grid: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Backward copy for the body; forward-warped paint only for clothing within 6 px outside the body; holes
    filled from neighbours. Returns (RGBA, shares of the body that are mirror-copied and holes)."""
    bout, bgot, body, mirrored = backward(rgb, alpha, b, grid)
    fwd, _ = warp(rgb, alpha, field, np.zeros_like(alpha))
    band = np.asarray(Image.fromarray(body.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(13))) > 127
    under = (fwd[..., 3] > 0) & band & ~body
    out = np.where(under[..., None], fwd[..., :3], 0).astype(np.uint8)
    out[bgot] = bout[bgot]
    got = bgot | under
    holes = body & ~bgot
    n = max(1, int(body.sum()))
    if holes.any():
        filled, ok = fill_nearest(out, got, limit=holes | got, iters=60)
        out[holes & ok], got = filled[holes & ok], got | (holes & ok)
    return np.dstack([out, got.astype(np.uint8) * 255]), {"mirror": round(float(mirrored.sum() / n), 4),
                                                          "holes": round(float(holes.sum() / n), 4)}


def fit_rest(painted: Image.Image, guide_alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut the painted figure out and fit it onto the guide rest figure's box (height, feet, centre)."""
    fg, _ = ps.figure_mask(painted)
    b = ps.fig_bbox(fg)
    g = ps.fig_bbox(guide_alpha)
    crop = painted.crop(b)
    mask = Image.fromarray(fg[b[1]:b[3], b[0]:b[2]].astype(np.uint8) * 255)
    gh = g[3] - g[1]
    w = max(1, round(crop.width * gh / crop.height))
    crop, mask = crop.resize((w, gh), Image.LANCZOS), mask.resize((w, gh), Image.NEAREST)
    rgb = Image.new("RGB", (CELL_W, CELL_H), ps.KEY)
    alpha = Image.new("L", (CELL_W, CELL_H), 0)
    at = ((g[0] + g[2]) // 2 - w // 2, g[1])
    rgb.paste(crop, at)
    alpha.paste(mask, at)
    return despill(np.asarray(rgb).copy(), np.asarray(alpha) > 127)


def on_key(rgba: np.ndarray) -> Image.Image:
    img = Image.new("RGB", (CELL_W, CELL_H), ps.KEY)
    img.paste(Image.fromarray(rgba[..., :3]), (0, 0), Image.fromarray(rgba[..., 3]))
    return img


def to_sprites(frames: list[np.ndarray]) -> list[Image.Image]:
    """All frames of a direction → SPRITE_H tall, cropped to their shared bounding box so motion is kept."""
    union = np.zeros((CELL_H, CELL_W), dtype=bool)
    for f in frames:
        union |= f[..., 3] > 0
    x0, y0, x1, y1 = ps.fig_bbox(union, min_px=1)
    scale = (SPRITE_H - 2) / (y1 - y0)
    w = max(1, round((x1 - x0) * scale))
    out = []
    for f in frames:
        crop = Image.fromarray(f).crop((x0, y0, x1, y1))
        crop = crop.resize((w, SPRITE_H - 2), Image.LANCZOS, reducing_gap=3.0)
        cell = Image.new("RGBA", (w + 2, SPRITE_H), (0, 0, 0, 0))
        cell.paste(crop, (1, 1))
        out.append(cell)
    return out


# ---------------------------------------------------------------- run


async def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        print("MODEL_API_KEY is not set. Run with: uv run --env-file .env python scripts/f01_rig_spike.py")
        return 2
    if not (GUIDES / "rig.npz").exists():
        print("Rig guides missing. Run first: uv run modal run modal_app.py::rig_spike")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    api = ps.Api(os.environ["MODEL_API_KEY"])
    t0 = time.perf_counter()
    rig = np.load(GUIDES / "rig.npz")
    anchors = json.loads((GUIDES / "anchors.json").read_text())
    boxes = {d: [f["box"] for f in sorted(anchors["frames"], key=lambda f: f["cell"]) if f["dir"] == d]
             for d in ps.DIRS}
    keys = {d: [f"{d}_{f['tag']}_{f['i']}" for f in sorted(anchors["frames"], key=lambda f: f["cell"])
                if f["dir"] == d] for d in ps.DIRS}
    sheets = {d: Image.open(GUIDES / f"sheet_{d}.png").convert("RGB") for d in ps.DIRS}
    masks = {d: np.asarray(Image.open(GUIDES / f"mask_{d}.png")) > 127 for d in ps.DIRS}
    desc = ps.BODIES[BODY]["desc"]

    ref = Image.open(ps.BODIES[BODY]["ref"]).convert("RGB")
    ref.thumbnail((2048, 2048), Image.LANCZOS)
    ids = {"ref": await api.upload(ps.png_bytes(ref), "rig_ref.png"),
           "turn": await api.upload((GUIDES / "turnaround.png").read_bytes(), "rig_turn.png")}
    card_png = await api.edit(ps.card_prompt(desc), [ids["ref"], ids["turn"]])
    card = Image.open(io.BytesIO(card_png)).convert("RGB")
    card.save(OUT / "card.png")
    ids["card"] = await api.upload(ps.png_bytes(card), "rig_card.png")

    report: dict = {"body": BODY, "dirs": {}}
    prior: list[str] = []
    rest: dict = {}
    for d in ps.DIRS:  # S → E → N, each later direction sees the earlier rest paintings
        cell = sheets[d].crop(tuple(boxes[d][0]))
        guide_rest = masks[d][ps.box_slice(boxes[d][0])]
        gid = await api.upload(ps.png_bytes(ps.on_canvas([cell], ps.KEY)), f"rig_guide_{d}.png")
        prompt, refs = rest_prompt(desc, d, len(prior)), [ids["ref"], ids["card"], *prior, gid]
        pngs = await asyncio.gather(*(api.edit(prompt, refs) for _ in range(CANDIDATES)))
        cands = []
        for k, png in enumerate(pngs):
            img = ps.load_png(png, ps.SHEET)
            img.save(OUT / f"rest_{d}_cand{k}.png")
            fg, _ = ps.figure_mask(img)
            cands.append((norm_iou(fg, guide_rest), k, img))
        best_iou, best_k, best = max(cands, key=lambda c: c[0])
        check = await rest_check(api, ps.small_png(ps.png_bytes(ps.on_canvas([cell], ps.KEY))),
                                 ps.small_png(ps.png_bytes(best)), ps.FACING[d])
        best.save(OUT / f"rest_{d}.png")
        prior.append(await api.upload(ps.png_bytes(best), f"rig_rest_{d}.png"))
        rest[d] = fit_rest(best, guide_rest)
        report["dirs"][d] = {"candidate_silhouette_iou": [round(c[0], 3) for c in cands], "chosen": best_k,
                             "rest_check": check}

    for d in ps.DIRS:
        rgb, alpha = rest[d]
        gyx, grid = rig[f"gyx_{d}"], tuple(int(v) for v in rig[f"grid_{d}"])
        sheet = Image.new("RGB", (ps.SHEET, ps.SHEET), ps.KEY)
        frames, holes, mirror = [], [], []
        for key, box in zip(keys[d], boxes[d], strict=True):
            field = grid_fields(gyx, grid, rig[f"map_{key}"])
            back = {part: rig[f"back_{part}_{key}"] for part in ("gyx", "src", "mode")}
            rgba, shares = compose(rgb, alpha, back, field, grid)
            holes.append(shares["holes"])
            mirror.append(shares["mirror"])
            frames.append(rgba)
            sheet.paste(on_key(rgba), tuple(box[:2]))
        sheet.save(OUT / f"sheet_{d}.png")
        sprites = to_sprites(frames)
        small = Image.new("RGBA", (sprites[0].width * 4, SPRITE_H * 3), (0, 0, 0, 0))
        for i, sp in enumerate(sprites):
            small.paste(sp, ((i % 4) * sp.width, (i // 4) * SPRITE_H))
        small.save(OUT / f"sheet_{d}_64.png")
        walk = [Image.alpha_composite(Image.new("RGBA", sp.size, (40, 40, 48, 255)), sp)
                .resize((sp.width * 4, sp.height * 4), Image.NEAREST) for sp in sprites[4:]]
        walk[0].save(OUT / f"walk_{d}_64.gif", save_all=True, append_images=walk[1:], duration=110, loop=0)
        report["dirs"][d]["mirror"] = mirror
        report["dirs"][d]["holes"] = holes
        report["dirs"][d]["_sheet"] = ps.png_bytes(sheet)

    checks = await asyncio.gather(*(api.pose_check(ps.small_png(ps.png_bytes(sheets[d])),
                                                   ps.small_png(report["dirs"][d]["_sheet"]), ps.FACING[d])
                                    for d in ps.DIRS), return_exceptions=True)
    for d, c in zip(ps.DIRS, checks, strict=True):
        report["dirs"][d]["pose"] = {"error": repr(c)} if isinstance(c, BaseException) else c
    what = "three 4×3 sprite sheets of the same character facing the viewer, right, and away"
    try:
        painted = [ps.small_png(report["dirs"][d].pop("_sheet")) for d in ps.DIRS]
        report["spark"] = await api.judge(ps.small_png(ps.png_bytes(card)), painted, what)
    except Exception as e:  # noqa: BLE001 - the spike reports every failure
        report["spark"] = {"error": repr(e)}
    report["cost_usd"], report["calls"] = round(api.meter.cost, 3), api.meter.calls
    report["seconds"] = round(time.perf_counter() - t0, 1)

    ok_all = report["spark"].get("same_character", 0) >= ps.MIN_SAME
    for d in ps.DIRS:
        r = report["dirs"][d]
        pose, rc = r["pose"], r["rest_check"]
        ok = ((pose.get("match") or 0) >= ps.MIN_POSE and (pose.get("facing") or 0) >= ps.MIN_FACING
              and max(r["holes"]) <= MAX_HOLES)
        ok_all = ok_all and ok
        print(f"[{'OK' if ok else '--'}] {d}: rest pose {'match' if rc['pose_match'] else 'OFF'} "
              f"(silhouettes {r['candidate_silhouette_iou']}, kept #{r['chosen']}) · sheet pose "
              f"{pose.get('match')}/12 facing {pose.get('facing')}/12 · holes max {max(r['holes']):.1%} · "
              f"mirror-copied max {max(r['mirror']):.0%}")
    sp = report["spark"]
    print(f"[{'OK' if sp.get('same_character', 0) >= ps.MIN_SAME else '--'}] same character "
          f"{sp.get('same_character')} style {sp.get('style_match')}")
    report["pass"] = ok_all
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'PASS' if ok_all else 'NOT YET'} · cost ≈ ${report['cost_usd']} · {report['seconds']}s · "
          f"look at {OUT}/walk_s_64.gif, walk_e_64.gif, walk_n_64.gif")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
