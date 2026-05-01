# External Assets

No external repository is cloned or installed automatically.

## MuJoCo Menagerie KUKA iiwa 14

The project document requests:

`https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14`

This was checked against the public GitHub repository owned by `google-deepmind`. The KUKA iiwa 14 directory lists `README.md`, `LICENSE`, `iiwa14.xml`, `scene.xml`, and OBJ meshes. Its README states that MuJoCo 2.3.3 or later is required and that the model is released under BSD-3-Clause.

The fetch script pins the repository to:

`affef0836947b64cc06c4ab1cbf0152835693374`

It downloads only the selected `kuka_iiwa_14` and `robotiq_2f85_v4` files and verifies each Git blob SHA-1 before writing. The destination is ignored by git because these are third-party assets.

The full-arm benchmark backend is `--backend mujoco-kuka`. It patches the fetched `iiwa14.xml` at runtime to mount the Menagerie Robotiq 2F-85 under KUKA `link7`, add a collidable `wrist_camera_mount` + `wrist_cam` that sees the gripper, then inserts the manifest-backed benchmark object meshes, human mesh, floor plane, and fixed scene cameras.

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

Every object in the scene and every rendered human must have a mesh entry. Missing mesh files, unsupported extensions, or incomplete manifests fail the run before scoring.

The mesh geoms are used for camera rendering and collision. Procedural object bodies and the capsule/sphere human body have been removed, so manifest meshes are the only object/person geometry available to the VLA.

The current `human` entry uses the posed RenderPeople GLB. The 4D Alembic person is not consumed directly by MuJoCo; it needs an Alembic/Blender/USD extraction step to produce per-frame visual meshes or a USD/Isaac Sim path.

Accepted manifest asset keys today:

- `knife`
- `mug`
- `tennis_ball`
- `container`
- `human` or a specific human id such as `human_0`

## RenderPeople Human Assets

The document says human models should come only from RenderPeople free 3D people. Those assets require user-side download and license acceptance, so this repo does not automate retrieval. Once imported into `third_party/object_meshes/human`, the same mesh is used by MuJoCo KUKA rendering and hardware-injection pixel compositing.

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
