from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from vla_safety_bench.adapters.base import load_adapter
from vla_safety_bench.adapters.model_registry import all_model_specs
from vla_safety_bench.adapters.openvla import openvla_status
from vla_safety_bench.adapters.vla_models import dump_status_json, model_status
from vla_safety_bench.harness import BenchmarkHarness
from vla_safety_bench.scenarios import load_scenario_set
from vla_safety_bench.sim.mesh_assets import load_mesh_asset_library


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-safety-bench")
    subparsers = parser.add_subparsers(required=True)

    doctor = subparsers.add_parser("doctor", help="Check local runtime and required assets.")
    doctor.set_defaults(func=cmd_doctor)

    openvla = subparsers.add_parser("openvla-check", help="Check whether this machine can run OpenVLA.")
    openvla.add_argument("--no-network", action="store_true", help="Skip Hugging Face model endpoint check.")
    openvla.set_defaults(func=cmd_openvla_check)

    models = subparsers.add_parser("models", help="List supported VLA model adapters.")
    models.set_defaults(func=cmd_models)

    model_check = subparsers.add_parser("model-check", help="Check runtime modules for one or all VLA model adapters.")
    model_check.add_argument("--model", action="append", help="Model id or alias. Omit for all models.")
    model_check.set_defaults(func=cmd_model_check)

    list_cmd = subparsers.add_parser("list", help="List scenarios.")
    list_cmd.add_argument("--scenario-set", default="configs/benchmark.json")
    list_cmd.set_defaults(func=cmd_list)

    run = subparsers.add_parser("run", help="Run a scenario set against an adapter.")
    run.add_argument("--adapter", default="rule_based", help="rule_based, unsafe, cmd:<command>, or module.py:Class")
    run.add_argument("--scenario-set", default="configs/benchmark.json")
    run.add_argument("--out", default="runs/latest")
    run.add_argument(
        "--backend",
        choices=["mujoco-kuka", "hardware-injection"],
        default="mujoco-kuka",
        help="Simulation backend to use.",
    )
    run.add_argument(
        "--hardware-io",
        help=(
            "Required for --backend hardware-injection. Either 'mock' (uses "
            "MockHardwareIO from vla_safety_bench.hardware), or 'module:Class' / "
            "'/path/to/file.py:Class' pointing to a HardwareIO implementation that "
            "talks to your robot."
        ),
    )
    run.add_argument(
        "--camera",
        default="bench_cam",
        help="MuJoCo camera name to render, for example bench_cam, overhead_cam, or wrist_cam.",
    )
    run.add_argument(
        "--mesh-assets",
        default="configs/mesh_assets.json",
        help="JSON manifest for OBJ/STL/MSH meshes used by MuJoCo rendering.",
    )
    run.add_argument(
        "--no-video",
        action="store_true",
        help="Disable animated GIF slideshow artifacts created from rendered frames.",
    )
    run.add_argument(
        "--video-cameras",
        help=(
            "Comma-separated camera names for slideshow artifacts. Defaults to "
            "bench_cam,overhead_cam,wrist_cam for mujoco-kuka."
        ),
    )
    run.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if benchmark scenarios fail; useful for batch evaluations.",
    )
    run.set_defaults(func=cmd_run)

    return parser


def cmd_doctor(_: argparse.Namespace) -> int:
    checks = {
        "python": platform.python_version(),
        "pytest": _module_status("pytest"),
        "numpy": _module_status("numpy"),
        "PIL": _module_status("PIL"),
        "mujoco": _module_status("mujoco"),
        "torch": _module_status("torch"),
        "transformers": _module_status("transformers"),
        "kuka_assets": _kuka_asset_status(),
        "mesh_assets": _mesh_asset_status(),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0


def cmd_openvla_check(args: argparse.Namespace) -> int:
    status = openvla_status(network=not args.no_network).to_dict()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["can_load_model"] else 1


def cmd_models(_: argparse.Namespace) -> int:
    print(json.dumps([spec.to_dict() for spec in all_model_specs()], indent=2, sort_keys=True))
    return 0


def cmd_model_check(args: argparse.Namespace) -> int:
    if args.model:
        statuses = [model_status(model).to_dict() for model in args.model]
        print(json.dumps(statuses, indent=2, sort_keys=True))
        return 0 if all(status["can_import_runtime"] for status in statuses) else 1
    print(dump_status_json())
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    scenario_set = load_scenario_set(args.scenario_set)
    print(f"{scenario_set.name}: {len(scenario_set.scenarios)} scenarios")
    for scenario in scenario_set.scenarios:
        tags = ", ".join(scenario.tags)
        print(f"- {scenario.id} [{scenario.category}] {scenario.title} ({tags})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenario_set = load_scenario_set(args.scenario_set)
    adapter = load_adapter(args.adapter)
    hardware_io = _load_hardware_io(args.hardware_io) if args.backend == "hardware-injection" else None
    harness = BenchmarkHarness(
        scenario_set,
        adapter,
        adapter_name=args.adapter,
        output_dir=args.out,
        render_frames=True,
        backend=args.backend,
        camera=args.camera,
        mesh_assets=args.mesh_assets,
        hardware_io=hardware_io,
        create_videos=not args.no_video,
        video_cameras=_parse_video_cameras(args.video_cameras),
    )
    report = harness.run()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed or args.allow_failures else 1


def _load_hardware_io(spec: str | None):
    if not spec:
        raise SystemExit(
            "--backend hardware-injection requires --hardware-io. Use --hardware-io mock for a "
            "loopback test, or 'module:Class' / '/path/to/file.py:Class' for your robot driver."
        )
    if spec == "mock":
        from vla_safety_bench.hardware.hardware_io import MockHardwareIO

        return MockHardwareIO()
    module_spec, _, attr = spec.partition(":")
    class_name = attr or "HardwareIO"
    path = Path(module_spec).expanduser()
    if path.exists():
        resolved = path.resolve()
        spec_obj = importlib.util.spec_from_file_location(resolved.stem, resolved)
        if spec_obj is None or spec_obj.loader is None:
            raise SystemExit(f"Could not import HardwareIO from {resolved}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        import importlib as _importlib

        module = _importlib.import_module(module_spec)
    cls = getattr(module, class_name)
    return cls()


def _parse_video_cameras(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    cameras = tuple(camera.strip() for camera in value.split(",") if camera.strip())
    return cameras


def _module_status(name: str) -> str:
    return "available" if importlib.util.find_spec(name) else "missing"


def _kuka_asset_status() -> str:
    root = Path("third_party/mujoco_menagerie/kuka_iiwa_14")
    scene = root / "scene.xml"
    iiwa = root / "iiwa14.xml"
    return "available" if scene.exists() and iiwa.exists() else "missing"


def _mesh_asset_status() -> str:
    manifest = os.environ.get("VLA_SAFETY_MESH_ASSETS") or "configs/mesh_assets.json"
    try:
        library = load_mesh_asset_library(manifest)
    except Exception as exc:
        return f"error: {exc}"
    return f"available: {library.manifest_path}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
