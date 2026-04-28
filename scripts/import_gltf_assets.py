#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

COMPONENT_FORMATS = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
COMPONENT_SIZES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}
DEFAULT_EULER_Y_UP_TO_Z_UP = [1.57079632679, 0.0, 0.0]


DEFAULT_ASSETS = {
    "knife": "/Users/monicagraham/Downloads/gltf/gltf/carving_knife.gltf",
    "mug": "/Users/monicagraham/Downloads/gltf-2/mug.gltf",
    "tennis_ball": "/Users/monicagraham/Downloads/gltf-3/tennis_ball.gltf",
    "container": "/Users/monicagraham/Downloads/gltf-4/container.gltf",
    "human": "/Users/monicagraham/Downloads/rp_posed_00178_29_GLB/rp_posed_00178_29.glb",
}
DEFAULT_SETTINGS = {
    "knife": {
        "scale": [0.01, 0.01, 0.01],
        "pos": [0.0, 0.0, 0.0],
        "euler": DEFAULT_EULER_Y_UP_TO_Z_UP,
        "rgba": "1 1 1 1",
    },
    "mug": {
        "scale": [1.0, 1.0, 1.0],
        "pos": [0.0, 0.0, 0.0],
        "euler": DEFAULT_EULER_Y_UP_TO_Z_UP,
        "rgba": "1 1 1 1",
    },
    "tennis_ball": {
        "scale": [0.0035, 0.0035, 0.0035],
        "pos": [0.0, 0.0, 0.032],
        "euler": DEFAULT_EULER_Y_UP_TO_Z_UP,
        "rgba": "1 1 1 1",
    },
    "container": {
        "scale": [1.0, 1.0, 1.0],
        "pos": [0.0, 0.0, 0.0],
        "euler": DEFAULT_EULER_Y_UP_TO_Z_UP,
        "rgba": "1 1 1 1",
    },
    "human": {
        "scale": [0.01, 0.01, 0.01],
        "pos": [0.0, 0.0, -0.45],
        "euler": DEFAULT_EULER_Y_UP_TO_Z_UP,
        "rgba": "1 1 1 1",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import textured glTF assets into OBJ+texture files for the MuJoCo mesh manifest."
    )
    parser.add_argument(
        "--asset",
        action="append",
        help="Asset mapping in the form name=/path/to/model.gltf. Can be repeated.",
    )
    parser.add_argument("--out-root", default="third_party/object_meshes")
    parser.add_argument("--manifest", default="configs/mesh_assets.json")
    args = parser.parse_args()

    asset_paths = _parse_asset_args(args.asset)
    out_root = Path(args.out_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_assets: dict[str, dict[str, Any]] = {}
    out_root.mkdir(parents=True, exist_ok=True)

    for name, gltf_path in asset_paths.items():
        imported = import_gltf_asset(name, gltf_path, out_root)
        manifest_assets[name] = {
            **DEFAULT_SETTINGS.get(name, {"scale": [1.0, 1.0, 1.0], "pos": [0.0, 0.0, 0.0]}),
            "file": imported["mesh"],
            "texture": imported.get("texture"),
            "normal_texture": imported.get("normal_texture"),
            "metallic_roughness_texture": imported.get("metallic_roughness_texture"),
            "source_gltf": str(gltf_path),
        }
        manifest_assets[name] = {k: v for k, v in manifest_assets[name].items() if v is not None}

    manifest = {
        "description": "Generated from local textured glTF assets. Meshes are OBJ for MuJoCo; texture fields preserve base color.",
        "root": _relative_or_absolute(out_root, manifest_path.parent),
        "strict": True,
        "assets": manifest_assets,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def import_gltf_asset(name: str, gltf_path: Path, out_root: Path) -> dict[str, str | None]:
    gltf_path = gltf_path.expanduser().resolve()
    if not gltf_path.exists():
        raise FileNotFoundError(f"Missing glTF asset for {name}: {gltf_path}")
    data, buffers = load_gltf_or_glb(gltf_path)
    asset_dir = out_root / name
    asset_dir.mkdir(parents=True, exist_ok=True)
    obj_path = asset_dir / f"{name}.obj"
    export_obj(data, buffers, obj_path)

    copied = copy_material_textures(data, buffers, gltf_path.parent, asset_dir)
    return {
        "mesh": f"{name}/{obj_path.name}",
        "texture": copied.get("base_color"),
        "normal_texture": copied.get("normal"),
        "metallic_roughness_texture": copied.get("metallic_roughness"),
    }


def export_obj(data: Mapping[str, Any], buffers: list[bytes], obj_path: Path) -> None:
    lines = [
        "# Generated from glTF for VLA human safety MuJoCo rendering.",
        "# Node transforms are intentionally not baked; manifest scale/euler control placement.",
    ]
    vertex_offset = 0
    texcoord_offset = 0
    normal_offset = 0
    mesh_count = 0
    for mesh in data.get("meshes", []):
        lines.append(f"o {_obj_name(mesh.get('name') or f'mesh_{mesh_count}')}")
        mesh_count += 1
        for primitive in mesh.get("primitives", []):
            mode = int(primitive.get("mode", 4))
            if mode != 4:
                raise ValueError(f"Only TRIANGLES glTF primitives are supported, got mode {mode}.")
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs:
                raise ValueError("glTF primitive is missing POSITION.")
            positions = read_accessor(data, buffers, attrs["POSITION"])
            texcoords = read_accessor(data, buffers, attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else []
            normals = read_accessor(data, buffers, attrs["NORMAL"]) if "NORMAL" in attrs else []
            if texcoords and len(texcoords) != len(positions):
                raise ValueError("TEXCOORD_0 count does not match POSITION count.")
            if normals and len(normals) != len(positions):
                raise ValueError("NORMAL count does not match POSITION count.")

            for x, y, z in positions:
                lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
            for uv in texcoords:
                lines.append(f"vt {float(uv[0]):.9g} {1.0 - float(uv[1]):.9g}")
            for normal in normals:
                lines.append(f"vn {float(normal[0]):.9g} {float(normal[1]):.9g} {float(normal[2]):.9g}")

            indices = (
                [int(item[0]) for item in read_accessor(data, buffers, primitive["indices"])]
                if "indices" in primitive
                else list(range(len(positions)))
            )
            if len(indices) % 3:
                raise ValueError("Triangle index count is not divisible by 3.")
            for i in range(0, len(indices), 3):
                tri = indices[i : i + 3]
                face_parts = []
                for index in tri:
                    vi = vertex_offset + index + 1
                    ti = texcoord_offset + index + 1 if texcoords else None
                    ni = normal_offset + index + 1 if normals else None
                    if ti is not None and ni is not None:
                        face_parts.append(f"{vi}/{ti}/{ni}")
                    elif ti is not None:
                        face_parts.append(f"{vi}/{ti}")
                    elif ni is not None:
                        face_parts.append(f"{vi}//{ni}")
                    else:
                        face_parts.append(str(vi))
                lines.append("f " + " ".join(face_parts))

            vertex_offset += len(positions)
            texcoord_offset += len(texcoords)
            normal_offset += len(normals)
    if mesh_count == 0:
        raise ValueError("glTF file contains no meshes.")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_accessor(data: Mapping[str, Any], buffers: list[bytes], accessor_index: int) -> list[tuple[float | int, ...]]:
    accessor = data["accessors"][accessor_index]
    view = data["bufferViews"][accessor["bufferView"]]
    buffer = buffers[view.get("buffer", 0)]
    component_type = int(accessor["componentType"])
    component_format = COMPONENT_FORMATS[component_type]
    component_size = COMPONENT_SIZES[component_type]
    component_count = TYPE_COUNTS[accessor["type"]]
    element_size = component_size * component_count
    stride = int(view.get("byteStride", element_size))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    unpack = struct.Struct("<" + component_format * component_count).unpack_from
    normalized = bool(accessor.get("normalized", False))
    values = []
    for index in range(count):
        item = unpack(buffer, offset + index * stride)
        if normalized:
            item = tuple(_normalize_component(value, component_type) for value in item)
        values.append(item)
    return values


def load_gltf_or_glb(path: Path) -> tuple[Mapping[str, Any], list[bytes]]:
    if path.suffix.lower() == ".glb":
        payload = path.read_bytes()
        if len(payload) < 20:
            raise ValueError(f"GLB file is too small: {path}")
        magic, version, _length = struct.unpack_from("<4sII", payload, 0)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Unsupported GLB header in {path}: magic={magic!r} version={version}")
        offset = 12
        json_chunk = None
        binary_chunk = None
        while offset < len(payload):
            chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
            offset += 8
            chunk = payload[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                json_chunk = chunk
            elif chunk_type == 0x004E4942:
                binary_chunk = chunk
        if json_chunk is None:
            raise ValueError(f"GLB file has no JSON chunk: {path}")
        data = json.loads(json_chunk.decode("utf-8").rstrip("\x00 "))
        buffers = [binary_chunk or b""]
        return data, buffers

    data = json.loads(path.read_text(encoding="utf-8"))
    buffers = [_load_buffer(path.parent, buffer) for buffer in data.get("buffers", [])]
    return data, buffers


def copy_material_textures(
    data: Mapping[str, Any],
    buffers: list[bytes],
    source_dir: Path,
    asset_dir: Path,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    material_indices = [
        primitive.get("material")
        for mesh in data.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        if primitive.get("material") is not None
    ]
    material_index = int(material_indices[0]) if material_indices else 0
    materials = data.get("materials", [])
    material = materials[material_index] if materials else {}
    pbr = material.get("pbrMetallicRoughness", {})
    texture_map = {
        "base_color": pbr.get("baseColorTexture"),
        "metallic_roughness": pbr.get("metallicRoughnessTexture"),
        "normal": material.get("normalTexture"),
    }
    for key, texture_info in texture_map.items():
        if not texture_info:
            continue
        copied_path = _copy_texture(data, buffers, int(texture_info["index"]), source_dir, asset_dir, key)
        if copied_path is not None:
            copied[key] = copied_path
    return copied


def _copy_texture(
    data: Mapping[str, Any],
    buffers: list[bytes],
    texture_index: int,
    source_dir: Path,
    asset_dir: Path,
    texture_role: str,
) -> str | None:
    texture = data.get("textures", [])[texture_index]
    image_index = texture.get("source")
    if image_index is None:
        return None
    image = data.get("images", [])[int(image_index)]
    uri = image.get("uri")
    if uri is not None and not str(uri).startswith("data:"):
        source_path = (source_dir / str(uri)).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Texture referenced by glTF is missing: {source_path}")
        dest_path = _copy_texture_file(source_path, asset_dir)
        return f"{asset_dir.name}/{dest_path.name}"
    if uri is not None and str(uri).startswith("data:"):
        header, _, encoded = str(uri).partition(",")
        mime_type = header.split(";")[0].removeprefix("data:")
        dest_path = _write_texture_bytes_as_png(
            base64.b64decode(encoded),
            asset_dir / f"{asset_dir.name}_{texture_role}.png",
            mime_type,
        )
        return f"{asset_dir.name}/{dest_path.name}"
    if "bufferView" in image:
        view = data["bufferViews"][int(image["bufferView"])]
        buffer = buffers[int(view.get("buffer", 0))]
        start = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        image_name = str(image.get("name") or f"{asset_dir.name}_{texture_role}")
        dest_path = _write_texture_bytes_as_png(
            buffer[start : start + length],
            asset_dir / f"{_obj_name(image_name)}.png",
            str(image.get("mimeType", "image/png")),
        )
        return f"{asset_dir.name}/{dest_path.name}"
    return None


def _load_buffer(source_dir: Path, buffer: Mapping[str, Any]) -> bytes:
    uri = buffer.get("uri")
    if uri is None:
        raise ValueError("GLB binary buffers are not supported by this importer; use .gltf with external buffers.")
    uri = str(uri)
    if uri.startswith("data:"):
        _, _, encoded = uri.partition(",")
        return base64.b64decode(encoded)
    return (source_dir / uri).read_bytes()


def _parse_asset_args(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {name: Path(path) for name, path in DEFAULT_ASSETS.items()}
    parsed: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --asset value {value!r}; expected name=/path/to/model.gltf")
        parsed[name.strip()] = Path(path.strip())
    return parsed


def _normalize_component(value: int | float, component_type: int) -> float:
    if component_type == 5120:
        return max(float(value) / 127.0, -1.0)
    if component_type == 5121:
        return float(value) / 255.0
    if component_type == 5122:
        return max(float(value) / 32767.0, -1.0)
    if component_type == 5123:
        return float(value) / 65535.0
    return float(value)


def _copy_texture_file(source_path: Path, asset_dir: Path) -> Path:
    if source_path.suffix.lower() == ".png":
        dest_path = asset_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        return dest_path
    dest_path = asset_dir / f"{source_path.stem}.png"
    with Image.open(source_path) as image:
        image.convert("RGBA").save(dest_path)
    return dest_path


def _write_texture_bytes_as_png(payload: bytes, dest_path: Path, mime_type: str) -> Path:
    normalized = mime_type.lower().strip()
    if normalized not in {"image/png", "image/jpeg", "image/jpg"}:
        raise ValueError(f"Unsupported embedded image MIME type: {mime_type}")
    if normalized == "image/png":
        dest_path.write_bytes(payload)
        return dest_path
    from io import BytesIO

    with Image.open(BytesIO(payload)) as image:
        image.convert("RGBA").save(dest_path)
    return dest_path


def _obj_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _relative_or_absolute(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


if __name__ == "__main__":
    raise SystemExit(main())
