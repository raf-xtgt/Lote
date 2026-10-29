"""Day-1 Meta Model API check (setup 01, step 5).

Answers the open questions in ARCHITECTURE.md > Capability Check before any feature depends on them:
access from our region, Muse Image pixel size and key-color background, SAM 3.1 on generated art,
Spark strict JSON output, Muse Wonder computer tool with truncation="auto".

Run from backend/:  uv run --env-file .env python scripts/day1_meta_check.py
Outputs go to backend/runs/day1/ (gitignored). Expected cost: about $0.10.
"""

import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from meta_sam_parser import (
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from openai import AsyncOpenAI
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
OUT = Path(__file__).resolve().parent.parent / "runs" / "day1"
KEY_RGB = (255, 0, 255)  # flat magenta key background (L-031)
N_IMAGES = 5

IMAGE_PROMPT = (
    "A single small knight character for a top-down 2D game, three-quarter view, full body, centered, "
    "painterly style, on a perfectly flat solid magenta (#FF00FF) background with no shadow, no gradient, "
    "no floor, no border."
)

results: dict[str, dict] = {}


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def report(name: str, ok: bool, **info) -> None:
    results[name] = {"ok": ok, **info}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {json.dumps(info, default=str)}")


async def check_spark(client: AsyncOpenAI) -> None:
    schema = {
        "type": "object",
        "properties": {
            "assets": {"type": "array", "items": {"type": "string"}},
            "tile_px": {"type": "integer"},
        },
        "required": ["assets", "tile_px"],
        "additionalProperties": False,
    }
    t0 = time.perf_counter()
    try:
        r = await client.responses.create(
            model="muse-spark-1.3",
            input="List 3 assets a tiny dungeon game needs and a tile size in pixels.",
            text={"format": {"type": "json_schema", "name": "Kit", "schema": schema, "strict": True}},
        )
        parsed = json.loads(r.output_text)
        ok = isinstance(parsed.get("assets"), list) and isinstance(parsed.get("tile_px"), int)
        report("spark_structured", ok, seconds=round(time.perf_counter() - t0, 1), output=parsed,
               usage=r.usage.model_dump() if r.usage else None)
    except Exception as e:  # noqa: BLE001 - day-1 probe reports every failure
        report("spark_structured", False, error=repr(e))


async def one_image(client: AsyncOpenAI, i: int) -> bytes | None:
    t0 = time.perf_counter()
    try:
        r = await client.images.generate(
            model="muse-image-1.0",
            prompt=IMAGE_PROMPT,
            size="1024x1024",
            output_format="png",
            extra_body={
                "reasoning_strength": "low",
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False,
                                    "enable_shell": False},
            },
        )
        png = base64.b64decode(r.data[0].b64_json)
        (OUT / f"image_{i}.png").write_bytes(png)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        arr = np.asarray(img).astype(int)
        # How clean is the key background? Share of pixels within 40 of magenta, and the corner colors.
        near_key = (np.abs(arr - np.array(KEY_RGB)).max(axis=2) <= 40).mean()
        corners = [arr[0, 0].tolist(), arr[0, -1].tolist(), arr[-1, 0].tolist(), arr[-1, -1].tolist()]
        report(f"image_{i}", True, seconds=round(time.perf_counter() - t0, 1), size=img.size,
               format=r.output_format, near_key_share=round(float(near_key), 3), corners=corners)
        return png
    except Exception as e:  # noqa: BLE001
        report(f"image_{i}", False, seconds=round(time.perf_counter() - t0, 1), error=repr(e))
        return None


async def check_sam(client: AsyncOpenAI, png: bytes) -> None:
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(png))
        w, h = img.size
        source = await client.responses.create(
            model="sam-3.1",
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "knight"},
                {"type": "input_image", "image_url": data_url(png)},
            ]}],
            metadata={"mask_encoding": "one_bit"},
            stream=True,
        )
        async with parse_responses_stream(source, image_segmentation_format()) as stream:
            async for _ in stream:
                pass
            result = await stream.final_result()
        masks = [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
        full = np.zeros((h, w), dtype=bool)
        for m in masks:
            raster = np.frombuffer(decode_mask_to_raster(m.mask), dtype=np.uint8)
            local = Image.fromarray(raster.reshape(m.mask.height, m.mask.width) * 255)
            b = m.bounds
            box = (int(b.left), int(b.top), int(b.right), int(b.bottom))
            local = local.resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST)
            full[box[1]:box[3], box[0]:box[2]] |= np.asarray(local) > 0
        Image.fromarray(full.astype(np.uint8) * 255).save(OUT / "sam_mask_0.png")
        # Compare with the key-color mask (the IoU check from L-031).
        arr = np.asarray(img.convert("RGB")).astype(int)
        key_mask = np.abs(arr - np.array(KEY_RGB)).max(axis=2) > 40
        union = (key_mask | full).sum()
        iou = float((key_mask & full).sum() / union) if union else 0.0
        report("sam_on_generated", len(masks) > 0, seconds=round(time.perf_counter() - t0, 1),
               objects=len(masks), mask_coverage=round(float(full.mean()), 3),
               key_vs_sam_iou=round(iou, 3), diagnostics=len(result.diagnostics))
    except Exception as e:  # noqa: BLE001
        report("sam_on_generated", False, seconds=round(time.perf_counter() - t0, 1), error=repr(e))


async def check_wonder_truncation(client: AsyncOpenAI, png: bytes) -> None:
    t0 = time.perf_counter()
    try:
        r = await client.responses.create(
            model="muse-wonder-1.0",
            tools=[{"type": "computer"}],
            truncation="auto",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "This is a game screen. Click the knight."},
                {"type": "input_image", "image_url": data_url(png)},
            ]}],
        )
        kinds = [o.type for o in r.output]
        report("wonder_computer_truncation_auto", True, seconds=round(time.perf_counter() - t0, 1),
               output_types=kinds, usage=r.usage.model_dump() if r.usage else None)
    except Exception as e:  # noqa: BLE001
        report("wonder_computer_truncation_auto", False, seconds=round(time.perf_counter() - t0, 1),
               error=repr(e))


async def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        print("MODEL_API_KEY is not set. Run with: uv run --env-file .env python scripts/day1_meta_check.py")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
    t0 = time.perf_counter()

    await check_spark(client)
    images = await asyncio.gather(*(one_image(client, i) for i in range(N_IMAGES)))
    first = next((p for p in images if p), None)
    if first is None:
        report("sam_on_generated", False, error="no image to segment")
        report("wonder_computer_truncation_auto", False, error="no image to send")
    else:
        await asyncio.gather(check_sam(client, first), check_wonder_truncation(client, first))

    sizes = {tuple(v["size"]) for k, v in results.items() if k.startswith("image_") and v["ok"]}
    print(f"\nImage sizes seen: {sorted(sizes)}")
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    failed = [k for k, v in results.items() if not v["ok"]]
    print(f"Saved to {OUT}. Failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
