from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import quoteattr

from vla_safety_bench.types import HumanState, ObjectState, Vector3, vector3

SUPPORTED_MESH_SUFFIXES = {".obj", ".stl", ".msh"}


@dataclass(frozen=True)
class MeshAssetSpec:
    name: str
    file_path: Path
    scale: Vector3 = (1.0, 1.0, 1.0)
    pos: Vector3 = (0.0, 0.0, 0.0)
    euler: Vector3 = (0.0, 0.0, 0.0)
    rgba: str | None = None
    material: str | None = None
    texture_path: Path | None = None
    normal_texture_path: Path | None = None
    metallic_roughness_texture_path: Path | None = None

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any], root: Path) -> "MeshAssetSpec":
        if "file" not in data:
            raise ValueError(f"Mesh asset {name!r} is missing required field 'file'.")
        file_path = Path(str(data["file"])).expanduser()
        if not file_path.is_absolute():
            file_path = root / file_path
        file_path = file_path.resolve()
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_MESH_SUFFIXES:
            raise ValueError(
                f"Mesh asset {name!r} uses unsupported file type {suffix!r}; "
                "use OBJ, STL, or MuJoCo MSH."
            )
        if not file_path.exists():
            raise FileNotFoundError(f"Mesh asset {name!r} is missing at {file_path}.")
        return cls(
            name=name,
            file_path=file_path,
            scale=vector3(data.get("scale"), default=(1.0, 1.0, 1.0)),
            pos=vector3(data.get("pos"), default=(0.0, 0.0, 0.0)),
            euler=vector3(data.get("euler"), default=(0.0, 0.0, 0.0)),
            rgba=_optional_text(data.get("rgba")),
            material=_optional_text(data.get("material")),
            texture_path=_resolve_optional_asset_path(data.get("texture"), root, name, "texture"),
            normal_texture_path=_resolve_optional_asset_path(
                data.get("normal_texture"), root, name, "normal_texture"
            ),
            metallic_roughness_texture_path=_resolve_optional_asset_path(
                data.get("metallic_roughness_texture"), root, name, "metallic_roughness_texture"
            ),
        )

    @property
    def mesh_name(self) -> str:
        return f"mesh_{_xml_name(self.name)}"

    @property
    def texture_name(self) -> str:
        return f"tex_{_xml_name(self.name)}_basecolor"

    @property
    def material_name(self) -> str:
        return self.material or f"mat_{_xml_name(self.name)}_visual"

    def asset_xml(self) -> str:
        entries: list[str] = []
        if self.texture_path is not None:
            entries.append(
                f'    <texture name={quoteattr(self.texture_name)} type="2d" '
                f'file={quoteattr(str(self.texture_path))}/>'
            )
            entries.append(
                f'    <material name={quoteattr(self.material_name)} '
                f'texture={quoteattr(self.texture_name)} rgba={quoteattr(self.rgba or "1 1 1 1")}/>'
            )
        entries.append(
            f'    <mesh name={quoteattr(self.mesh_name)} file={quoteattr(str(self.file_path))} '
            f'scale={quoteattr(_vec(self.scale))}/>'
        )
        return "\n".join(entries)

    def geom_xml(
        self,
        geom_name: str,
        *,
        visual_only: bool = False,
        mass: float | None = None,
        friction: str | None = None,
    ) -> str:
        material_attr = f" material={quoteattr(self.material_name)}" if self.material or self.texture_path else ""
        rgba_attr = f" rgba={quoteattr(self.rgba)}" if self.rgba and not material_attr else ""
        contact_attr = ' contype="0" conaffinity="0"' if visual_only else ""
        mass_attr = f' mass="{mass:g}"' if mass is not None else ""
        friction_attr = f" friction={quoteattr(friction)}" if friction else ""
        return (
            f'      <geom name={quoteattr(geom_name)} type="mesh" mesh={quoteattr(self.mesh_name)}'
            f' pos={quoteattr(_vec(self.pos))} euler={quoteattr(_vec(self.euler))}'
            f"{material_attr}{rgba_attr}{contact_attr}{mass_attr}{friction_attr}/>"
        )

    def visual_geom_xml(self, geom_name: str) -> str:
        return self.geom_xml(geom_name, visual_only=True)


@dataclass(frozen=True)
class MeshAssetLibrary:
    manifest_path: Path
    strict: bool
    assets: dict[str, MeshAssetSpec]

    @classmethod
    def from_file(cls, path: str | Path) -> "MeshAssetLibrary":
        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Mesh asset manifest is missing: {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Mesh asset manifest root must be a JSON object.")
        root_value = data.get("root", ".")
        root = Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = manifest_path.parent / root
        root = root.resolve()
        raw_assets = data.get("assets")
        if not isinstance(raw_assets, Mapping):
            raise ValueError("Mesh asset manifest must include an 'assets' object.")
        assets = {
            str(name): MeshAssetSpec.from_dict(str(name), spec, root)
            for name, spec in raw_assets.items()
            if isinstance(spec, Mapping)
        }
        if len(assets) != len(raw_assets):
            raise ValueError("Each mesh asset manifest entry must be a JSON object.")
        return cls(
            manifest_path=manifest_path,
            strict=True,
            assets=assets,
        )

    def mesh_asset_xml(self) -> str:
        return "\n".join(spec.asset_xml() for spec in self.assets.values())

    def object_spec(self, name: str) -> MeshAssetSpec | None:
        return self.assets.get(name)

    def human_spec(self, human: HumanState) -> MeshAssetSpec | None:
        return self.assets.get(human.id) or self.assets.get("human")

    def table_spec(self) -> MeshAssetSpec | None:
        return self.assets.get("table")

    def validate_scene(self, objects: Iterable[ObjectState], humans: Iterable[HumanState]) -> None:
        missing: list[str] = []
        for obj in objects:
            if self.object_spec(obj.name) is None:
                missing.append(obj.name)
        for human in humans:
            if self.human_spec(human) is None:
                missing.append(human.id if human.id in self.assets else "human")
        if missing:
            raise ValueError(
                f"Mesh asset manifest {self.manifest_path} is missing required scene meshes: "
                f"{', '.join(sorted(set(missing)))}"
            )


def load_mesh_asset_library(
    path: MeshAssetLibrary | str | Path | None = None,
    *,
    required: bool = False,
) -> MeshAssetLibrary | None:
    if isinstance(path, MeshAssetLibrary):
        return path
    manifest = path or os.environ.get("VLA_SAFETY_MESH_ASSETS")
    if not manifest:
        if required:
            raise ValueError(
                "MuJoCo rendering requires a mesh asset manifest. Pass --mesh-assets or set "
                "VLA_SAFETY_MESH_ASSETS."
            )
        return None
    return MeshAssetLibrary.from_file(manifest)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_optional_asset_path(value: Any, root: Path, asset_name: str, field: str) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh asset {asset_name!r} {field} is missing at {path}.")
    return path


def _vec(value: Vector3) -> str:
    return f"{value[0]} {value[1]} {value[2]}"


def _xml_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
