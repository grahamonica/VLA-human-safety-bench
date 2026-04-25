from __future__ import annotations

from pathlib import Path

from vla_safety_bench.scenarios import ScenarioSpec
from vla_safety_bench.types import HumanState, ObjectState

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None


class SyntheticOverlayRenderer:
    """Renders lightweight 2D human overlays until approved 3D assets exist."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

    @property
    def available(self) -> bool:
        return Image is not None and ImageDraw is not None

    def render(
        self,
        scenario: ScenarioSpec,
        step_index: int,
        humans: list[HumanState],
        objects: list[ObjectState],
        output_dir: Path,
    ) -> str | None:
        if not self.available:
            return None
        frame_dir = output_dir / "frames" / scenario.id
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{step_index:03d}.png"

        image = Image.new("RGB", (self.width, self.height), (238, 240, 242))
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_workspace(draw)
        self._draw_objects(draw, objects)
        for human in humans:
            self._draw_human(draw, human)
        draw.rectangle((0, 0, self.width - 1, self.height - 1), outline=(50, 55, 60, 255), width=1)
        draw.text((12, 12), f"{scenario.id} step {step_index}", fill=(30, 35, 40, 255))
        image.save(frame_path)
        return str(frame_path)

    def _project(self, x_m: float, y_m: float) -> tuple[int, int]:
        # Robot base at image center. One meter maps to 150 px in this synthetic camera.
        return (int(self.width / 2 + y_m * 150), int(self.height * 0.70 - x_m * 150))

    def _draw_workspace(self, draw: "ImageDraw.ImageDraw") -> None:
        cx, cy = self._project(0.0, 0.0)
        radius = 120
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(255, 95, 85, 32),
            outline=(200, 60, 50, 120),
            width=3,
        )
        draw.rectangle((140, 260, 500, 420), fill=(185, 190, 196, 255), outline=(120, 125, 132, 255))
        draw.rectangle((300, 310, 340, 350), fill=(240, 126, 40, 255), outline=(100, 80, 60, 255))

    def _draw_objects(self, draw: "ImageDraw.ImageDraw", objects: list[ObjectState]) -> None:
        for obj in objects:
            px, py = self._project(obj.position_m[0], obj.position_m[1])
            if obj.name == "knife":
                draw.rectangle((px - 24, py - 3, px + 24, py + 3), fill=(70, 74, 78, 255))
                draw.polygon([(px + 24, py - 7), (px + 42, py), (px + 24, py + 7)], fill=(170, 175, 180, 255))
            elif obj.name == "mug":
                draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill=(58, 128, 190, 255))
                draw.arc((px + 9, py - 8, px + 26, py + 8), 270, 90, fill=(58, 128, 190, 255), width=4)
            elif obj.name == "fork":
                draw.rectangle((px - 18, py - 2, px + 18, py + 2), fill=(120, 125, 130, 255))
                for tine in (-5, 0, 5):
                    draw.line((px + 18, py + tine, px + 30, py + tine), fill=(120, 125, 130, 255), width=2)
            else:
                draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=(90, 110, 130, 255))

    def _draw_human(self, draw: "ImageDraw.ImageDraw", human: HumanState) -> None:
        px, py = self._project(human.position_m[0], human.position_m[1])
        fill = (215, 65, 60, 210) if human.in_danger_zone else (55, 140, 100, 210)
        outline = (130, 20, 20, 255) if human.in_danger_zone else (20, 90, 55, 255)
        draw.ellipse((px - 14, py - 64, px + 14, py - 36), fill=fill, outline=outline, width=2)
        draw.rounded_rectangle((px - 18, py - 34, px + 18, py + 44), radius=10, fill=fill, outline=outline, width=2)
        draw.line((px - 18, py - 10, px - 42, py + 16), fill=outline, width=5)
        draw.line((px + 18, py - 10, px + 42, py + 16), fill=outline, width=5)
        draw.line((px - 8, py + 44, px - 20, py + 84), fill=outline, width=5)
        draw.line((px + 8, py + 44, px + 20, py + 84), fill=outline, width=5)

