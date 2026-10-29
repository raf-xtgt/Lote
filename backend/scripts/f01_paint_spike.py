"""F-01 slice 1 spike, round 3: paint frames one per call or four per call, anchored on a painted idle frame.

Rounds 1–2 (one call per 12-frame sheet) kept the character but not the poses or the grid (BUILD_LOG L-072,
L-073). Round 3, per body (spec.md > Wiring Blueprint 1; features/01-characters/spike-results.md):
  anchors: idle frame 0 per direction, painted alone, S → E → N, each seeing the directions already painted
  frame (option 2): every other frame painted alone, seeing the card + that direction's anchor
  row4 (option 1): four frames per call on a 2×2 canvas, seeing the card + that direction's anchor
Painted figures are cut out by bounding box and fitted onto the guide figure's box (as Align will), giving the
same 4×3 sheets as round 2. Metrics: cells kept (a figure, not spilling out of its region), raw size spread,
Spark per-cell pose + facing check, Spark consistency score; SAM 12-box check on the frame method's S sheet.

Run from backend/:  uv run --env-file .env python scripts/f01_paint_spike.py
Needs runs/spike/guides/ first. Writes runs/spike/paint/ and runs/spike/report.json. Cost about $1.1.
"""

import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    SegmentationMaskRecord,
    build_object_prompt,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from openai import AsyncOpenAI
from PIL import Image, ImageFilter

BASE_URL = "https://api.meta.ai/v1"
HERE = Path(__file__).resolve().parent.parent
GUIDES = HERE / "runs" / "spike" / "guides"
OUT = HERE / "runs" / "spike" / "paint"
REPORT = HERE / "runs" / "spike" / "report.json"
REFS = HERE / "tests" / "fixtures" / "refs"

IMAGE_MODEL, SPARK_MODEL, SAM_MODEL = "muse-image-1.0", "muse-spark-1.3", "sam-3.1"
KEY = (255, 0, 255)
SHEET, FRAMES = 1600, 12  # 4 columns × 3 rows of 400×533 cells; cell boxes come from anchors.json
DIRS = ("s", "e", "n")
NO_TOOLS = {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}
# ARCHITECTURE.md > AI Components
PRICE = {"image": 0.01, "sam": 0.0025, "spark_in": 1.25e-6, "spark_out": 4.25e-6}

# Pass rule (spec.md > How We'll Verify > Spike, revised in L-072): per sheet ≥ 11/12 cells kept (a figure,
# inside its cell), Spark says ≥ 9/12 poses match the guide and ≥ 11/12 face the right way, figure heights
# within 20% of each other; Spark same-character ≥ 8. Silhouette IoU is reported only: it can't tell poses
# apart.
MIN_CELLS, MIN_POSE, MIN_FACING, MAX_SIZE_SPREAD, MIN_SAME = 11, 9, 11, 1.2, 8

BODIES = {
    "humanoid": {
        "ref": REFS / "humanoid" / "humanoid.jpeg",
        "desc": "the armored knight from the reference (steel helmet with nose guard, chainmail, plate "
                "armor, blue and gold tabard), carrying no weapon or shield",
    },
    "quadruped": {
        "ref": REFS / "creature" / "creature.jpeg",
        "desc": "the dark red dragon from the reference, walking on four legs, wings folded tight against "
                "its back",
    },
}
FACING = {"s": "facing the viewer (front)", "e": "facing right (side view)", "n": "facing away (back view)"}

BG_RULE = ("Keep the background perfectly flat solid magenta (#FF00FF): no shadows, ground, scenery, "
           "borders, grid lines or text.")


def card_prompt(desc: str) -> str:
    return (f"Image 1 is the style and character reference. Image 2 shows three gray clay figures on flat "
            f"magenta: the character facing the viewer, facing right, and facing away. Paint {desc} over "
            f"each gray figure, in the art style of image 1 (its line work, shading, colors and level of "
            f"detail). Keep each figure's exact pose, size, position and outline. Three figures, nothing "
            f"else. {BG_RULE}")


POSE_RULE = ("copy the gray figure's exact pose: which leg is forward, how far each leg and arm swings, how "
             "the body leans, which way it faces. Keep its position and size.")


def anchor_prompt(desc: str, d: str, n_prior: int) -> str:
    prior = ""
    if n_prior == 1:
        prior = "Image 3 is the same character already painted facing another direction; match it. "
    elif n_prior > 1:
        prior = (f"Images 3 to {2 + n_prior} are the same character already painted facing other directions; "
                 f"match them. ")
    return (f"Image 1 is the style reference. Image 2 is the design card of the character: {desc}. {prior}"
            f"The last image is one gray clay figure on flat magenta, {FACING[d]}, standing idle. Paint the "
            f"character from the design card over the gray figure, in the style of image 1, and {POSE_RULE} "
            f"One figure only. {BG_RULE}")


def frame_prompt(desc: str, d: str) -> str:
    return (f"Image 1 is the style reference. Image 2 is the design card of the character: {desc}. Image 3 "
            f"is this character already painted {FACING[d]}: match it exactly (colors, details, proportions, "
            f"size). The last image is one gray clay figure on flat magenta: one animation frame, "
            f"{FACING[d]}. Paint the same character over the gray figure and {POSE_RULE} One figure only. "
            f"{BG_RULE}")


def row_prompt(desc: str, d: str) -> str:
    return (f"Image 1 is the style reference. Image 2 is the design card of the character: {desc}. Image 3 "
            f"is this character already painted {FACING[d]}: match it exactly (colors, details, proportions, "
            f"size). The last image is 4 gray clay figures on flat magenta, one in each quarter of the "
            f"image: 4 animation frames, {FACING[d]}. Paint the same character over each gray figure; each "
            f"figure stays inside its own quarter. For each one, {POSE_RULE} Four figures only. {BG_RULE}")


# ---------------------------------------------------------------- images and metrics


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def load_png(data: bytes, size: int | None = None) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def small_png(png: bytes, side: int = 1024) -> bytes:
    im = Image.open(io.BytesIO(png))
    im.thumbnail((side, side))
    return png_bytes(im)


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def figure_mask(img: Image.Image) -> tuple[np.ndarray, list[int]]:
    """Foreground = far from the background color, estimated from the image border (the painter may shift
    it)."""
    a = np.asarray(img).astype(int)
    border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3),
                             a[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    return np.abs(a - bg).max(axis=2) > 48, [int(x) for x in bg]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def box_slice(box: list[int]) -> tuple[slice, slice]:
    return slice(box[1], box[3]), slice(box[0], box[2])


def fig_height(m: np.ndarray) -> int:
    b = fig_bbox(m)
    return b[3] - b[1] if b else 0


def fig_bbox(m: np.ndarray, min_px: int = 3) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of rows/columns with at least `min_px` foreground pixels (ignores specks)."""
    rows, cols = np.nonzero(m.sum(axis=1) >= min_px)[0], np.nonzero(m.sum(axis=0) >= min_px)[0]
    if not len(rows) or not len(cols):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


# Guide frames go to the painter enlarged: one frame fills most of the 1600² canvas; four frames sit one per
# 800² quarter. Painted figures are cut out by bounding box and fitted back onto the guide figure's box
# (as spec.md > Align does), so both methods are measured on the same 4×3 sheets as round 2.
SINGLE_SCALE, SINGLE_AT = 2.4, (320, 160)
QUAD_SCALE, QUAD_AT = 1.2, (160, 80)


def on_canvas(cells: list[Image.Image], key: tuple[int, int, int]) -> Image.Image:
    """1 cell → centred on 1600²; 4 cells → one per 800² quarter."""
    canvas = Image.new("RGB", (SHEET, SHEET), key)
    if len(cells) == 1:
        c = cells[0]
        canvas.paste(c.resize((round(c.width * SINGLE_SCALE), round(c.height * SINGLE_SCALE)), Image.LANCZOS),
                     SINGLE_AT)
        return canvas
    for q, c in enumerate(cells):
        x, y = (q % 2) * SHEET // 2 + QUAD_AT[0], (q // 2) * SHEET // 2 + QUAD_AT[1]
        canvas.paste(c.resize((round(c.width * QUAD_SCALE), round(c.height * QUAD_SCALE)), Image.LANCZOS),
                     (x, y))
    return canvas


def regions(n: int) -> list[tuple[int, int, int, int]]:
    if n == 1:
        return [(0, 0, SHEET, SHEET)]
    h = SHEET // 2
    return [((q % 2) * h, (q // 2) * h, (q % 2 + 1) * h, (q // 2 + 1) * h) for q in range(4)]


def cut_out(painted: Image.Image, region: tuple[int, int, int, int]) -> dict:
    """The painted figure inside one region: its crop, height, and whether it spills over the region's rim."""
    fg, _ = figure_mask(painted)
    m = fg[region[1]:region[3], region[0]:region[2]]
    rim = np.concatenate([m[:4].ravel(), m[-4:].ravel(), m[:, :4].ravel(), m[:, -4:].ravel()])
    b = fig_bbox(m)
    if b is None or m.mean() < 0.01:
        return {"crop": None, "height": 0, "spills": False}
    crop = painted.crop((region[0] + b[0], region[1] + b[1], region[0] + b[2], region[1] + b[3]))
    return {"crop": crop, "height": b[3] - b[1], "spills": bool(rim.mean() > 0.02)}


def fit_sheet(cuts: list[dict], guide_alpha: Image.Image, boxes: list[list[int]],
              key: tuple[int, int, int]) -> Image.Image:
    """Each cut-out scaled to its guide figure's height, feet on the guide's feet, centred on the guide."""
    guide = np.asarray(guide_alpha) > 127
    sheet = Image.new("RGB", (SHEET, SHEET), key)
    for cut, box in zip(cuts, boxes, strict=True):
        g = fig_bbox(guide[box_slice(box)])
        if cut["crop"] is None or g is None:
            continue
        gh = g[3] - g[1]
        c = cut["crop"]
        w = max(1, round(c.width * gh / c.height))
        c = c.resize((w, gh), Image.LANCZOS)
        cx = box[0] + (g[0] + g[2]) // 2
        cell = sheet.crop(tuple(box))
        cell.paste(c, (cx - box[0] - w // 2, g[1]))
        sheet.paste(cell, tuple(box[:2]))
    return sheet


def measure(cuts: list[dict], guide_scale: list[float], fitted: Image.Image,
            guide_alpha: Image.Image) -> dict:
    """Kept = a figure was painted and it stays inside its region. Size spread uses the raw painted heights
    relative to each guide figure (before fitting)."""
    kept = [i for i, c in enumerate(cuts) if c["crop"] is not None and not c["spills"]]
    rel = [cuts[i]["height"] / guide_scale[i] for i in kept if guide_scale[i]]
    return {"cells_kept": len(kept), "cells_lost": [i for i in range(len(cuts)) if i not in kept],
            "size_spread": round(max(rel) / max(1e-6, min(rel)), 2) if rel else None,
            "key_kept": key_kept(fitted, np.asarray(guide_alpha) > 127)}


def key_kept(painted: Image.Image, guide: np.ndarray) -> float:
    """Share of pixels away from any guide figure that are still within 48 of the pure key color."""
    near = Image.fromarray(guide.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(25))
    away = np.asarray(near) == 0
    a = np.asarray(painted).astype(int)
    close = np.abs(a - np.array(KEY)).max(axis=2) <= 48
    return round(float(close[away].mean()), 3)


# ---------------------------------------------------------------- API calls


class Meter:
    def __init__(self) -> None:
        self.cost, self.calls = 0.0, {"image": 0, "spark": 0, "sam": 0, "files": 0}

    def add(self, kind: str, cost: float) -> None:
        self.calls[kind] += 1
        self.cost += cost


class Api:
    def __init__(self, key: str) -> None:
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=key, max_retries=3, timeout=600)
        self.meter = Meter()
        self.image_slots = asyncio.Semaphore(12)

    async def upload(self, png: bytes, name: str) -> str:
        f = await self.client.files.create(file=(name, png, "image/png"), purpose="user_data",
                                           expires_after={"anchor": "created_at", "seconds": 7 * 86400})
        self.meter.add("files", 0.0)
        return f.id

    async def edit(self, prompt: str, file_ids: list[str]) -> bytes:
        """`images/edits` with the Meta JSON body (`images[{file_id}]`). The SDK's `images.edit` only sends
        multipart, so this goes through its generic `post` (same auth, retries on 429 / 5xx)."""
        body = {"model": IMAGE_MODEL, "prompt": prompt, "images": [{"file_id": f} for f in file_ids], "n": 1,
                "size": "1024x1024", "output_format": "png", "tool_enablement": NO_TOOLS}
        async with self.image_slots:
            r = await self.client.post("/images/edits", body=body, cast_to=object)
        self.meter.add("image", PRICE["image"])
        return base64.b64decode(r["data"][0]["b64_json"])

    async def turn(self, content: list[dict], previous: str | None) -> tuple[bytes, str]:
        """One Muse Image Responses turn → (png, response id)."""
        extra = {"previous_response_id": previous} if previous else {}
        async with self.image_slots:
            r = await self.client.responses.create(
                model=IMAGE_MODEL, input=[{"role": "user", "content": content}],
                tools=[{"type": "image_generation", **NO_TOOLS, "size": "1024x1024", "output_format": "png"}],
                **extra)
        self.meter.add("image", PRICE["image"])
        b64 = next(o.result for o in r.output if o.type == "image_generation_call")
        return base64.b64decode(b64), r.id

    async def judge(self, card: bytes, images: list[bytes], what: str) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "same_character": {"type": "integer", "description": "0-10: all figures are one character"},
                "style_match": {"type": "integer", "description": "0-10: matches the card's style"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["same_character", "style_match", "issues", "summary"],
            "additionalProperties": False,
        }
        ask = (f"Image 1 is a character design card. The other images are {what}. Judge whether every figure "
               "is the same character as the card (same body, colors, outfit, details) and in the same art "
               "style. Ignore the flat magenta background. List concrete issues with the image and cell "
               "(row, column).")
        content = [{"type": "input_text", "text": ask},
                   {"type": "input_image", "image_url": data_url(card)}]
        content += [{"type": "input_image", "image_url": data_url(p)} for p in images]
        return await self.spark(content, "Consistency", schema)

    async def spark(self, content: list[dict], name: str, schema: dict) -> dict:
        r = await self.client.responses.create(
            model=SPARK_MODEL, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}})
        u = r.usage
        self.meter.add("spark", (u.input_tokens * PRICE["spark_in"] + u.output_tokens * PRICE["spark_out"])
                       if u else 0.0)
        return json.loads(r.output_text)

    async def pose_check(self, guide: bytes, painted: bytes, facing: str) -> dict:
        """Spark compares each painted cell with its gray guide cell: same pose? facing the right way?"""
        cell = {"type": "object",
                "properties": {"cell": {"type": "integer"}, "pose_match": {"type": "boolean"},
                               "facing_ok": {"type": "boolean"}},
                "required": ["cell", "pose_match", "facing_ok"], "additionalProperties": False}
        schema = {"type": "object",
                  "properties": {"cells": {"type": "array", "items": cell}, "notes": {"type": "string"}},
                  "required": ["cells", "notes"], "additionalProperties": False}
        ask = ("Image 1 is a guide sheet: 12 gray clay figures in a grid of 4 columns × 3 rows. Image 2 is "
               "the same grid painted as a character. Cells are numbered 0 to 11, left to right, top to "
               "bottom. For every cell: pose_match is true only if the painted figure has the gray figure's "
               "pose: the same leg forward, similar leg and arm swing, the same lean. Compare limb "
               "positions, not looks; extra bulk from armor, wings or tail doesn't matter. facing_ok is true "
               f"if it faces {facing} like the gray figure. Be strict. Return all 12 cells; notes names the "
               "cells that fail and why.")
        content = [{"type": "input_text", "text": ask},
                   {"type": "input_image", "image_url": data_url(guide)},
                   {"type": "input_image", "image_url": data_url(painted)}]
        r = await self.spark(content, "PoseCheck", schema)
        cells = {c["cell"]: c for c in r["cells"] if 0 <= c["cell"] < FRAMES}
        return {"match": sum(c["pose_match"] for c in cells.values()),
                "facing": sum(c["facing_ok"] for c in cells.values()),
                "failed": sorted(i for i, c in cells.items() if not (c["pose_match"] and c["facing_ok"])),
                "cells_answered": len(cells), "notes": r["notes"]}

    async def sam_boxes(self, sheet: bytes, boxes: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
        """One SAM 3.1 request, one box per object → {object_id: full-size bool mask}."""
        prompt = build_object_prompt(size=(SHEET, SHEET), objects=[
            PromptObject(id=i + 1, prompts=[FramePrompt(box=PromptBox(*b))]) for i, b in enumerate(boxes)])
        source = await self.client.responses.create(
            model=SAM_MODEL, stream=True,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url(sheet)}]}])
        async with parse_responses_stream(source, image_segmentation_format()) as stream:
            async for _ in stream:
                pass
            result = await stream.final_result()
        self.meter.add("sam", PRICE["sam"])
        out: dict[str, np.ndarray] = {}
        for m in result.records:
            if not isinstance(m, SegmentationMaskRecord):
                continue
            raster = np.frombuffer(decode_mask_to_raster(m.mask), dtype=np.uint8)
            local = Image.fromarray(raster.reshape(m.mask.height, m.mask.width) * 255)
            b = m.bounds
            box = (int(b.left), int(b.top), int(b.right), int(b.bottom))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            local = local.resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST)
            full = out.setdefault(m.object_id, np.zeros((SHEET, SHEET), dtype=bool))
            full[box[1]:box[3], box[0]:box[2]] |= np.asarray(local) > 0
        return out
# ---------------------------------------------------------------- methods

METHODS = {"frame": "one frame per call (option 2)", "row4": "4 frames per call, 2×2 (option 1)"}


async def paint_anchors(api: Api, body: str, ids: dict, cells: dict, out: Path) -> dict:
    """Idle frame 0 per direction, S → E → N, each call seeing the directions already painted."""
    desc, prior, anchors = BODIES[body]["desc"], [], {}
    for d in DIRS:
        gid = await api.upload(png_bytes(on_canvas([cells[d][0]], KEY)), f"{body}_guide_{d}0.png")
        png = await api.edit(anchor_prompt(desc, d, len(prior)), [ids["ref"], ids["card"], *prior, gid])
        img = load_png(png, SHEET)
        img.save(out / f"anchor_{d}.png")
        aid = await api.upload(png_bytes(img), f"{body}_anchor_{d}.png")
        prior.append(aid)
        anchors[d] = {"img": img, "id": aid}
    return anchors


async def method_frame(api: Api, body: str, ids: dict, anchors: dict, cells: dict, out: Path) -> dict:
    desc = BODIES[body]["desc"]

    async def one(d: str, i: int) -> Image.Image:
        gid = await api.upload(png_bytes(on_canvas([cells[d][i]], KEY)), f"{body}_guide_{d}{i}.png")
        png = await api.edit(frame_prompt(desc, d), [ids["ref"], ids["card"], anchors[d]["id"], gid])
        img = load_png(png, SHEET)
        img.save(out / f"frame_{d}_{i:02}.png")
        return img

    jobs = [(d, i) for d in DIRS for i in range(1, FRAMES)]
    imgs = await asyncio.gather(*(one(d, i) for d, i in jobs))
    painted = {d: [anchors[d]["img"]] for d in DIRS}
    for (d, _), img in zip(jobs, imgs, strict=True):
        painted[d].append(img)
    return {d: [cut_out(img, regions(1)[0]) for img in painted[d]] for d in DIRS}


async def method_row4(api: Api, body: str, ids: dict, anchors: dict, cells: dict, out: Path) -> dict:
    desc = BODIES[body]["desc"]

    async def one(d: str, k: int) -> list[dict]:
        canvas = on_canvas(cells[d][4 * k:4 * k + 4], KEY)
        gid = await api.upload(png_bytes(canvas), f"{body}_guide_{d}_row{k}.png")
        png = await api.edit(row_prompt(desc, d), [ids["ref"], ids["card"], anchors[d]["id"], gid])
        img = load_png(png, SHEET)
        img.save(out / f"row4_{d}_{k}.png")
        return [cut_out(img, r) for r in regions(4)]

    jobs = [(d, k) for d in DIRS for k in range(3)]
    rows = await asyncio.gather(*(one(d, k) for d, k in jobs))
    cuts: dict = {d: [] for d in DIRS}
    for (d, _), r in zip(jobs, rows, strict=True):
        cuts[d] += r
    return cuts


async def sam_check(api: Api, sheet: bytes, boxes: list[list[int]]) -> dict:
    """Box prompts = guide cell boxes (inset 2 px) for the 12 frames, in one request."""
    inset = [(b[0] + 2, b[1] + 2, b[2] - 2, b[3] - 2) for b in boxes]
    t0 = time.perf_counter()
    masks = await api.sam_boxes(sheet, inset)
    fg, _ = figure_mask(load_png(sheet))
    ious = []
    for i, box in enumerate(boxes):
        m = masks.get(str(i + 1))
        sl = box_slice(box)
        ious.append(round(iou(fg[sl], m[sl]), 3) if m is not None else None)
    return {"boxes_sent": len(boxes), "objects_returned": len(masks), "key_vs_sam_iou": ious,
            "seconds": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------- per body


def summarize(res: dict) -> dict:
    return {"cells_kept": {d: res[d]["cells_kept"] for d in DIRS},
            "pose_match": {d: res[d].get("pose", {}).get("match") for d in DIRS},
            "facing_ok": {d: res[d].get("pose", {}).get("facing") for d in DIRS},
            "size_spread": {d: res[d]["size_spread"] for d in DIRS}}


async def run_body(api: Api, body: str) -> dict:
    g = GUIDES / body
    out = OUT / body
    out.mkdir(parents=True, exist_ok=True)
    guides = {f"{k}_{d}": Image.open(g / f"{k}_{d}.png") for k in ("sheet", "mask") for d in DIRS}
    anchors_json = json.loads((g / "anchors.json").read_text())
    boxes = {d: [f["box"] for f in sorted(anchors_json["frames"], key=lambda f: f["cell"]) if f["dir"] == d]
             for d in DIRS}
    cells = {d: [guides[f"sheet_{d}"].convert("RGB").crop(tuple(b)) for b in boxes[d]] for d in DIRS}
    guide_h = {d: [fig_height(np.asarray(guides[f"mask_{d}"])[box_slice(b)] > 127) for b in boxes[d]]
               for d in DIRS}

    ref = Image.open(BODIES[body]["ref"]).convert("RGB")
    ref.thumbnail((2048, 2048), Image.LANCZOS)  # model copy, as uploads will (spec.md > Upload validation)
    names = {"ref": png_bytes(ref), "turnaround": (g / "turnaround.png").read_bytes()}
    keys = list(names)
    uploaded = await asyncio.gather(*(api.upload(names[k], f"{body}_{k}.png") for k in keys))
    ids = dict(zip(keys, uploaded, strict=True))
    report: dict = {}

    t0 = time.perf_counter()
    card_png = await api.edit(card_prompt(BODIES[body]["desc"]), [ids["ref"], ids["turnaround"]])
    card = Image.open(io.BytesIO(card_png)).convert("RGB")
    card.save(out / "card.png")
    ids["card"] = await api.upload(png_bytes(card), f"{body}_card.png")
    card_b = small_png(png_bytes(card))
    anchors = await paint_anchors(api, body, ids, cells, out)
    report["card_and_anchors_seconds"] = round(time.perf_counter() - t0, 1)

    fns = {"frame": (method_frame, SINGLE_SCALE), "row4": (method_row4, QUAD_SCALE)}
    t1 = time.perf_counter()
    results = await asyncio.gather(*(fn(api, body, ids, anchors, cells, out) for fn, _ in fns.values()),
                                   return_exceptions=True)
    seconds = round(time.perf_counter() - t1, 1)
    for (name, (_, scale)), cuts in zip(fns.items(), results, strict=True):
        if isinstance(cuts, BaseException):
            report[name] = {"ok": False, "error": repr(cuts)}
            continue
        res: dict = {"ok": True, "seconds_both_methods": seconds}
        fitted = {}
        for d in DIRS:
            fitted[d] = fit_sheet(cuts[d], guides[f"mask_{d}"], boxes[d], KEY)
            fitted[d].save(out / f"{name}_{d}_sheet.png")
            res[d] = measure(cuts[d], [h * scale for h in guide_h[d]], fitted[d], guides[f"mask_{d}"])
        fitted_png = {d: png_bytes(fitted[d]) for d in DIRS}
        checks = await asyncio.gather(
            *(api.pose_check(small_png(png_bytes(guides[f"sheet_{d}"].convert("RGB"))),
                             small_png(fitted_png[d]), FACING[d]) for d in DIRS),
            return_exceptions=True)
        for d, pose in zip(DIRS, checks, strict=True):
            res[d]["pose"] = {"error": repr(pose)} if isinstance(pose, BaseException) else pose
        res["summary"] = summarize(res)
        what = "three 4×3 sprite sheets of the same character facing the viewer, right, and away"
        try:
            res["spark"] = await api.judge(card_b, [small_png(fitted_png[d]) for d in DIRS], what)
        except Exception as e:  # noqa: BLE001 - the spike reports every failure
            res["spark"] = {"error": repr(e)}
        if name == "frame":
            try:
                res["sam_boxes"] = await sam_check(api, fitted_png["s"], boxes["s"])
            except Exception as e:  # noqa: BLE001
                res["sam_boxes"] = {"error": repr(e)}
        report[name] = res
    return report


# ---------------------------------------------------------------- verdict


def method_passes(rep: dict, m: str) -> bool:
    r = rep.get(m, {})
    if not r.get("ok"):
        return False
    s = r["summary"]
    return r.get("spark", {}).get("same_character", 0) >= MIN_SAME and all(
        s["cells_kept"][d] >= MIN_CELLS and (s["pose_match"][d] or 0) >= MIN_POSE
        and (s["facing_ok"][d] or 0) >= MIN_FACING and (s["size_spread"][d] or 99) <= MAX_SIZE_SPREAD
        for d in DIRS)


def method_score(rep: dict, m: str) -> float:
    """Cells kept, poses matched (both per cell, 0–1) and Spark's two 0–10 scores (0–1), averaged."""
    r = rep[m]
    s, sp = r["summary"], r.get("spark", {})
    kept = statistics.mean(s["cells_kept"][d] for d in DIRS) / FRAMES
    pose = statistics.mean((s["pose_match"][d] or 0) for d in DIRS) / FRAMES
    spark = (sp.get("same_character", 0) + sp.get("style_match", 0)) / 20
    return round((kept + pose + spark) / 3, 3)


def verdict(bodies: dict) -> dict:
    ok = {m: all(method_passes(bodies[b], m) for b in bodies) for m in METHODS}
    scores = {m: round(statistics.mean(method_score(bodies[b], m) for b in bodies), 3)
              for m in METHODS if all(bodies[b].get(m, {}).get("ok") for b in bodies)}
    winners = [m for m in METHODS if ok[m]]
    if winners:
        winner = max(winners, key=lambda m: scores.get(m, 0))
        why = "passes cells kept, pose, facing, size and same-character for both bodies"
    else:
        winner, why = "none", "no method passed the rule; see per-body numbers and the PNGs"
    return {"winner": winner, "why": why, "passes": ok, "scores": scores, "names": METHODS}


async def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        print("MODEL_API_KEY is not set. Run with: uv run --env-file .env python scripts/f01_paint_spike.py")
        return 2
    stale = [b for b in BODIES if not (GUIDES / b / "anchors.json").exists()
             or "cols" not in json.loads((GUIDES / b / "anchors.json").read_text())]
    if stale:
        print(f"Guide sheets missing or old (4×4) for {stale}. "
              "Run first: uv run modal run modal_app.py::guide_spike")
        return 2
    api = Api(os.environ["MODEL_API_KEY"])
    t0 = time.perf_counter()
    results = await asyncio.gather(*(run_body(api, b) for b in BODIES), return_exceptions=True)
    bodies = {}
    for b, r in zip(BODIES, results, strict=True):
        bodies[b] = {"ok": False, "error": repr(r)} if isinstance(r, BaseException) else r
    report = {"round": 3, "bodies": bodies, "cost_usd": round(api.meter.cost, 3), "calls": api.meter.calls,
              "seconds": round(time.perf_counter() - t0, 1)}
    if all("frame" in bodies[b] for b in bodies):
        report["verdict"] = verdict(bodies)
    REPORT.write_text(json.dumps(report, indent=2, default=str))

    for b, rep in bodies.items():
        if "error" in rep and len(rep) == 2:
            print(f"[FAIL] {b}: {rep['error']}")
            continue
        for m in METHODS:
            r = rep.get(m, {})
            if not r.get("ok"):
                print(f"[FAIL] {b} {m}: {r.get('error')}")
                continue
            s, sp = r["summary"], r.get("spark", {})
            mark = "OK" if method_passes(rep, m) else "--"
            print(f"[{mark}] {b} {m}: kept {s['cells_kept']} pose {s['pose_match']} facing {s['facing_ok']} "
                  f"size {s['size_spread']} same {sp.get('same_character')} style {sp.get('style_match')}")
        seg = rep.get("frame", {}).get("sam_boxes", {})
        print(f"[{'OK' if seg.get('objects_returned') == FRAMES else 'FAIL'}] {b} SAM 12 boxes: "
              f"{seg.get('objects_returned', seg.get('error'))} objects")
    v = report.get("verdict", {})
    print(f"\nWinner: {v.get('winner')} ({v.get('names', {}).get(v.get('winner'), '')}): {v.get('why')}")
    print(f"Cost ≈ ${report['cost_usd']} · calls {report['calls']} · {report['seconds']}s · {REPORT}")
    return 0 if v.get("winner") not in (None, "none") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
