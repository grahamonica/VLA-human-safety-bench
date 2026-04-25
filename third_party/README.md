# Third-Party Assets

This directory is for local, explicitly fetched assets only. It is intentionally ignored by git except for this README.

Use:

```bash
python scripts/fetch_mujoco_kuka.py --dest third_party/mujoco_menagerie
```

The fetch script pins and verifies the MuJoCo Menagerie KUKA files. RenderPeople human assets must be added manually after license acceptance.

Object meshes should be placed under:

```text
third_party/object_meshes/
  knife/knife.obj
  mug/mug.obj
  tennis_ball/tennis_ball.obj
  container/container.obj
```

For the current textured glTF downloads, run:

```bash
python scripts/import_gltf_assets.py
```

That writes OBJ meshes, copies base-color/normal/roughness textures, and generates `configs/mesh_assets.json`. These files are intentionally ignored by git unless you decide to track vetted assets separately.
