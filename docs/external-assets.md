# External Assets

No external repository is cloned or installed automatically.

## MuJoCo Menagerie KUKA iiwa 14

The project document requests:

`https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14`

This was checked against the public GitHub repository owned by `google-deepmind`. The KUKA iiwa 14 directory lists `README.md`, `LICENSE`, `iiwa14.xml`, `scene.xml`, and OBJ meshes. Its README states that MuJoCo 2.3.3 or later is required and that the model is released under BSD-3-Clause.

The fetch script pins the repository to:

`affef0836947b64cc06c4ab1cbf0152835693374`

It downloads only the selected `kuka_iiwa_14` files and verifies each Git blob SHA-1 before writing. The destination is ignored by git because these are third-party assets.

The full-arm benchmark backend is `--backend mujoco-kuka`. It patches the fetched `iiwa14.xml` at runtime to add a `wrist_camera_mount` and `wrist_cam` under KUKA `link7`, then inserts the floor-mounted benchmark objects, human proxy, and fixed scene cameras.

## Object Meshes

The benchmark now supports real local OBJ, STL, or MuJoCo MSH meshes through a manifest. For the current textured glTF downloads, run:

```bash
python scripts/import_gltf_assets.py
```

The importer converts glTF/GLB geometry to OBJ with UVs, copies base-color/normal/roughness texture files, converts JPEG textures to PNG for MuJoCo, and writes `configs/mesh_assets.json`. Run MuJoCo with:

```bash
python -m vla_safety_bench run \
  --adapter rule_based \
  --scenario-set configs/smoke.json \
  --backend mujoco-kuka \
  --camera wrist_cam \
  --mesh-assets configs/mesh_assets.json \
  --out runs/kuka_real_mesh_smoke
```

When `strict` is true in the manifest, every object in the scene and the human proxy must have a mesh entry. Missing mesh files, unsupported extensions, or incomplete strict manifests fail the run before scoring.

The mesh geoms are used for camera rendering. The simulator keeps explicit primitive collision geoms for the knife, mug, tennis ball, container, and human body so physics contacts remain stable even when visual meshes are high-poly or decorative. If a mesh is present, those collision proxies are made transparent in the render.

The current `human` entry uses the posed RenderPeople GLB. The 4D Alembic person is not consumed directly by MuJoCo; it needs an Alembic/Blender/USD extraction step to produce per-frame visual meshes or a USD/Isaac Sim path.

Accepted manifest asset keys today:

- `knife`
- `mug`
- `tennis_ball`
- `container`
- `human` or a specific human id such as `human_0`

## Procedural Fallback Geometry

If no mesh manifest is provided, the knife, mug, tennis ball, container, and human proxy are procedural MuJoCo primitives defined in `vla_safety_bench/sim/mujoco_backend.py`. They are not external meshes or stock assets. This keeps the benchmark redistributable and avoids license ambiguity until approved object/human scans are added.

## RenderPeople Human Assets

The document says human models should come only from RenderPeople free 3D people. Those assets require user-side download and license acceptance, so this repo does not automate retrieval. Until approved assets are added, the benchmark uses synthetic 2D human overlays and a MuJoCo capsule proxy.

Suggested layout:

```text
third_party/object_meshes/
  knife/
    knife.obj
  mug/
    mug.obj
  tennis_ball/
    tennis_ball.obj
  container/
    container.obj
  human/
    human.obj
```
