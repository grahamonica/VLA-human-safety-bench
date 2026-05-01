#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_safety_bench.assets import (
    MENAGERIE_FILES,
    MENAGERIE_COMMIT,
    MENAGERIE_LICENSE,
    MENAGERIE_REPO,
    default_asset_root,
    git_blob_sha1,
    raw_menagerie_url,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd().resolve()
    dest = Path(args.dest).expanduser().resolve() if args.dest else default_asset_root(repo_root).resolve()
    _ensure_inside_repo(dest, repo_root)

    print(f"source=https://github.com/{MENAGERIE_REPO}")
    print(f"commit={MENAGERIE_COMMIT}")
    print(f"license={MENAGERIE_LICENSE}")
    print(f"dest={dest}")

    try:
        if args.verify_only:
            return verify_existing(dest)
        return fetch(dest, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch pinned MuJoCo Menagerie KUKA iiwa 14 + Robotiq 2F-85 assets."
    )
    parser.add_argument("--dest", default=None, help="Destination root. Default: third_party/mujoco_menagerie")
    parser.add_argument("--dry-run", action="store_true", help="Download and verify without writing files.")
    parser.add_argument("--verify-only", action="store_true", help="Verify files already present at destination.")
    return parser


def fetch(dest: Path, *, dry_run: bool = False) -> int:
    for relative_path, expected_sha in MENAGERIE_FILES.items():
        url = raw_menagerie_url(relative_path)
        data = _download(url)
        actual_sha = git_blob_sha1(data)
        if actual_sha != expected_sha:
            print(
                f"hash mismatch for {relative_path}: expected {expected_sha}, got {actual_sha}",
                file=sys.stderr,
            )
            return 1
        target = dest / relative_path
        if dry_run:
            print(f"verified {relative_path} ({len(data)} bytes)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"wrote {target}")
    return 0


def verify_existing(dest: Path) -> int:
    missing: list[str] = []
    mismatched: list[str] = []
    for relative_path, expected_sha in MENAGERIE_FILES.items():
        target = dest / relative_path
        if not target.exists():
            missing.append(relative_path)
            continue
        actual_sha = git_blob_sha1(target.read_bytes())
        if actual_sha != expected_sha:
            mismatched.append(relative_path)
    if missing or mismatched:
        for path in missing:
            print(f"missing {path}", file=sys.stderr)
        for path in mismatched:
            print(f"hash mismatch {path}", file=sys.stderr)
        return 1
    print(f"verified {len(MENAGERIE_FILES)} files")
    return 0


def _download(url: str) -> bytes:
    context = _ssl_context()
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except Exception:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _ensure_inside_repo(dest: Path, repo_root: Path) -> None:
    try:
        dest.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to write outside repository: {dest}") from exc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
