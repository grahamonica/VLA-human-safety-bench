from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from vla_safety_bench.types import JsonDict

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_SLIDE_DURATION_MS = 1000
LABEL_HEIGHT_PX = 28
MAX_GRID_COLUMNS = 3


@dataclass(frozen=True)
class VideoArtifact:
    scenario_id: str
    path: str
    format: str
    kind: str
    frame_count: int
    cameras: list[str]

    def to_dict(self) -> JsonDict:
        return {
            "scenario_id": self.scenario_id,
            "path": self.path,
            "format": self.format,
            "kind": self.kind,
            "frame_count": self.frame_count,
            "cameras": list(self.cameras),
        }


def camera_frame_path(output_dir: Path, scenario_id: str, camera: str, step_index: int) -> Path:
    return (
        output_dir
        / "frames"
        / scenario_id
        / "cameras"
        / _safe_camera_name(camera)
        / f"{step_index:03d}.png"
    )


def write_camera_slideshow(
    *,
    scenario_id: str,
    frames_by_step: Sequence[Mapping[str, str | Path]],
    output_path: Path,
    slide_duration_ms: int = DEFAULT_SLIDE_DURATION_MS,
) -> VideoArtifact | None:
    """Write an animated GIF that shows one simulation step per slide.

    Each slide is a grid of the available camera frames for that step. The model
    still receives its normal observation image; this function only creates a
    post-run review artifact.
    """

    if not frames_by_step:
        return None
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError(
            "Pillow is required to create simulation video slideshow artifacts."
        )

    camera_names = _ordered_camera_names(frames_by_step)
    if not camera_names:
        return None

    slides = [
        _compose_slide(step_index, camera_names, camera_frames)
        for step_index, camera_frames in enumerate(frames_by_step)
    ]
    if not slides:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    slides[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=slides[1:],
        duration=max(1, int(slide_duration_ms)),
        loop=0,
    )
    return VideoArtifact(
        scenario_id=scenario_id,
        path=str(output_path),
        format="gif",
        kind="multi_camera_slideshow",
        frame_count=len(slides),
        cameras=camera_names,
    )


def _ordered_camera_names(frames_by_step: Sequence[Mapping[str, str | Path]]) -> list[str]:
    cameras: list[str] = []
    seen: set[str] = set()
    for camera_frames in frames_by_step:
        for camera in camera_frames:
            if camera not in seen:
                cameras.append(camera)
                seen.add(camera)
    return cameras


def _compose_slide(step_index: int, camera_names: list[str], camera_frames: Mapping[str, str | Path]):
    opened = [_open_frame(camera_frames.get(camera)) for camera in camera_names]
    tile_width = max(image.width for image in opened)
    tile_height = max(image.height for image in opened)
    prepared = [_fit_to_tile(image, tile_width, tile_height) for image in opened]

    columns = min(MAX_GRID_COLUMNS, max(1, len(camera_names)))
    rows = math.ceil(len(camera_names) / columns)
    slide = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + LABEL_HEIGHT_PX)),
        (32, 35, 38),
    )
    draw = ImageDraw.Draw(slide)
    font = ImageFont.load_default()

    for index, (camera, image) in enumerate(zip(camera_names, prepared)):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + LABEL_HEIGHT_PX)
        draw.rectangle((x, y, x + tile_width, y + LABEL_HEIGHT_PX), fill=(24, 26, 28))
        draw.text((x + 10, y + 8), f"{camera} step {step_index}", fill=(240, 242, 244), font=font)
        slide.paste(image, (x, y + LABEL_HEIGHT_PX))

    return slide


def _open_frame(path: str | Path | None):
    if path is None:
        return _missing_frame()
    frame_path = Path(path)
    if not frame_path.exists():
        return _missing_frame()
    with Image.open(frame_path) as image:
        return image.convert("RGB").copy()


def _missing_frame():
    image = Image.new("RGB", (640, 480), (210, 214, 218))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "missing camera frame", fill=(42, 45, 48), font=ImageFont.load_default())
    return image


def _fit_to_tile(image, tile_width: int, tile_height: int):
    if image.width == tile_width and image.height == tile_height:
        return image
    fitted = Image.new("RGB", (tile_width, tile_height), (0, 0, 0))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    resized = image.resize((tile_width, tile_height), resampling)
    fitted.paste(resized, (0, 0))
    return fitted


def _safe_camera_name(camera: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in camera.strip()
    )
    return safe or "camera"
