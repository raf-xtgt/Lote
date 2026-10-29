"""Day-1 follow-up: find why the computer tool returned 400.

Round 1 (L-038): Wonder answers without tools; Wonder + computer → "Unsupported tool type: 'computer'";
Spark + computer → "native computer input is too large to store" with a 2.2 MB 1600x1600 PNG screenshot.
Round 2 (this version): a realistic 1280x720 JPEG screenshot, stored and stateless, Spark and Wonder.

Run from backend/:  uv run --env-file .env python scripts/day1_computer_probe.py [--wonder-only]
Builds the screenshot from runs/day1/image_0.png. Expected cost: under $0.05.
Failed calls print the UTC timestamp and request ID for a Meta support ticket.
"""

import asyncio
import base64
import io
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
SRC = Path(__file__).resolve().parent.parent / "runs" / "day1" / "image_0.png"
VIEWPORT = (1280, 720)  # playtest browser viewport


def screenshot_url() -> str:
    """Knight scaled onto a 1280x720 canvas, JPEG q80, like a real game screenshot."""
    knight = Image.open(SRC).convert("RGB").resize((360, 360), Image.LANCZOS)
    canvas = Image.new("RGB", VIEWPORT, (40, 44, 52))
    canvas.paste(knight, (460, 180))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=80)
    print(f"Screenshot: {VIEWPORT[0]}x{VIEWPORT[1]} JPEG, {len(buf.getvalue()) / 1024:.0f} KB")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def content(shot_url: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "input_text", "text": "This is a game screen. Click the knight."},
        {"type": "input_image", "image_url": shot_url},
    ]}]


async def probe(client: AsyncOpenAI, name: str, **kwargs) -> None:
    t0 = time.perf_counter()
    try:
        r = await client.responses.create(**kwargs)
        kinds = [o.type for o in r.output]
        print(f"[OK]   {name} ({time.perf_counter() - t0:.1f}s): output types {kinds}")
        for o in r.output:
            if o.type == "computer_call":
                print(f"       computer_call: {o.model_dump_json()[:400]}")
    except Exception as e:  # noqa: BLE001 - probe reports every failure
        print(f"[FAIL] {name} ({time.perf_counter() - t0:.1f}s): {e!r}")
        # What Meta support asks for: request ID and timestamp of each failed call (L-052).
        response = getattr(e, "response", None)
        headers = response.headers.items() if response is not None else []
        ids = {k: v for k, v in headers if "id" in k.lower()}
        print(f"       at {datetime.now(UTC).isoformat(timespec='seconds')}, "
              f"request_id {getattr(e, 'request_id', None)}, id headers {ids}")


async def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        print(f"MODEL_API_KEY is not set. Run with: uv run --env-file .env python {__file__}")
        return 2
    if not SRC.exists():
        print(f"Missing {SRC}; run day1_meta_check.py first.")
        return 2
    client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
    shot = screenshot_url()
    computer = [{"type": "computer"}]

    if "--wonder-only" not in sys.argv:
        await probe(client, "C spark + computer, stored", model="muse-spark-1.3", tools=computer,
                    input=content(shot))
        await probe(client, "D spark + computer + truncation auto", model="muse-spark-1.3", tools=computer,
                    truncation="auto", input=content(shot))
        await probe(client, "E spark + computer, stateless", model="muse-spark-1.3", tools=computer,
                    store=False, input=content(shot))
    await probe(client, "F wonder + computer, stored", model="muse-wonder-1.0", tools=computer,
                input=content(shot))
    await probe(client, "G wonder + computer, stateless", model="muse-wonder-1.0", tools=computer,
                store=False, input=content(shot))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
