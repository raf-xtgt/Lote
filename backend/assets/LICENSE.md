# Third-party assets

All files under `quaternius/` are by Quaternius, released under
[CC0 1.0 Universal (public domain)](https://creativecommons.org/publicdomain/zero/1.0/). No attribution required;
credited here anyway.

| File | Source pack | Use in Lote |
|---|---|---|
| `quaternius/characters/Superhero_Male_FullBody.gltf` + `.bin` | [Universal Base Characters](https://quaternius.com/packs/universalbasecharacters.html) (Standard, free), `Base Characters/Godot - UE/` | Rigged base body for character pose guides |
| `quaternius/characters/Superhero_Female_FullBody.gltf` + `.bin` | same | same |
| `quaternius/animations/UAL1_Standard.glb` | [Universal Animation Library](https://quaternius.com/packs/universalanimationlibrary.html) (Standard, free), `Unreal-Godot/` (no root motion) | Idle and walk clips for the guides |

Textures are left out on purpose: the guides render silhouette, depth and pose only, so the glTF image references
are not needed. The full downloads live in `quaternius/_download/` (gitignored).
