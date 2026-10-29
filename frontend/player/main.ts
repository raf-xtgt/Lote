// Phaser player entry (no React, L-021). Skeleton scene: one tile per key press on a 64 px grid (L-013, L-017).
import * as Phaser from "phaser";

const TILE = 64;

class SmokeScene extends Phaser.Scene {
  private player!: Phaser.GameObjects.Rectangle;

  create() {
    this.add.grid(0, 0, 640, 448, TILE, TILE, 0x1d2330, 1, 0x2e3648, 1).setOrigin(0);
    this.add.text(8, 8, "Lote player skeleton: arrow keys", { color: "#ffffff", fontSize: "18px" });
    this.player = this.add.rectangle(TILE * 2.5, TILE * 3.5, TILE - 8, TILE - 8, 0x4fa3ff);
    const step: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    this.input.keyboard?.on("keydown", (e: KeyboardEvent) => {
      const d = step[e.key];
      if (d) this.player.setPosition(this.player.x + d[0] * TILE, this.player.y + d[1] * TILE);
    });
  }
}

new Phaser.Game({
  type: Phaser.AUTO,
  parent: "game",
  width: 640,
  height: 448,
  pixelArt: true,
  scene: SmokeScene,
});
