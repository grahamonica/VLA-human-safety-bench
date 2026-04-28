import importlib.util

import pytest

from vla_safety_bench.video import camera_frame_path, write_camera_slideshow


def test_write_camera_slideshow_with_multiple_camera_frames(tmp_path):
    if importlib.util.find_spec("PIL") is None:
        pytest.skip("Pillow is not installed")

    from PIL import Image

    bench = tmp_path / "bench.png"
    overhead = tmp_path / "overhead.png"
    Image.new("RGB", (32, 24), (220, 20, 20)).save(bench)
    Image.new("RGB", (32, 24), (20, 80, 220)).save(overhead)

    artifact = write_camera_slideshow(
        scenario_id="scenario",
        frames_by_step=[{"bench_cam": bench, "overhead_cam": overhead}],
        output_path=tmp_path / "videos" / "scenario.gif",
    )

    assert artifact is not None
    assert artifact.cameras == ["bench_cam", "overhead_cam"]
    assert (tmp_path / "videos" / "scenario.gif").stat().st_size > 0


def test_camera_frame_path_sanitizes_camera_names(tmp_path):
    assert camera_frame_path(tmp_path, "scenario", "arm/wrist cam", 4) == (
        tmp_path / "frames" / "scenario" / "cameras" / "arm_wrist_cam" / "004.png"
    )
