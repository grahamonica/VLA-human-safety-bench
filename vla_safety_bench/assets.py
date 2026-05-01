from __future__ import annotations

import hashlib
from pathlib import Path

MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "affef0836947b64cc06c4ab1cbf0152835693374"
MENAGERIE_LICENSE = "BSD-3-Clause"

KUKA_IIWA_14_FILES = {
    "kuka_iiwa_14/CHANGELOG.md": "3366ea507a097a1e1bcf7e2e0f3f8ce0ead951a5",
    "kuka_iiwa_14/LICENSE": "d4681a468ef93f00712d31e79c5262f72b73299f",
    "kuka_iiwa_14/README.md": "f39aa704edd3e26f6de34ca75d92d78ebfb4d0eb",
    "kuka_iiwa_14/iiwa14.xml": "7ddda365d13af7abc2686046ad55f8018147528c",
    "kuka_iiwa_14/scene.xml": "25c4470d42b115ef648dd39525bb72d00d8a70b7",
    "kuka_iiwa_14/assets/band.obj": "d65fe23941215f0a667c022f763fb800fbdc5b79",
    "kuka_iiwa_14/assets/kuka.obj": "b0a7ab3d8fbdd8028006a3f039c020d466b217bd",
    "kuka_iiwa_14/assets/link_0.obj": "c40c530466b584a460a6edbec2e904c39cf697a0",
    "kuka_iiwa_14/assets/link_1.obj": "7dfc780791c56536da3bef8164bfa64d7be1d8a9",
    "kuka_iiwa_14/assets/link_2_grey.obj": "30f3016889842f980f2634d2152372152183780a",
    "kuka_iiwa_14/assets/link_2_orange.obj": "b2e0259f527df00081ff6ee9a7d01b823a9db489",
    "kuka_iiwa_14/assets/link_3.obj": "225ca9f05c284bedf2453ca61b251743081b5cde",
    "kuka_iiwa_14/assets/link_4_grey.obj": "88ea37e8e09aafdd1978e2cae398cd4c96c54e29",
    "kuka_iiwa_14/assets/link_4_orange.obj": "607b3f409600915fae45946862ce3b6ccb63ad7d",
    "kuka_iiwa_14/assets/link_5.obj": "55b239f50d9d7d358fdcfc375c5871a2a81cf07f",
    "kuka_iiwa_14/assets/link_6_grey.obj": "1695475f84082405c6b9161dabb6ecda21ac9f53",
    "kuka_iiwa_14/assets/link_6_orange.obj": "1903562ef70c28abfe175893fec2819e99e19e6b",
    "kuka_iiwa_14/assets/link_7.obj": "1f7848b1a78523d86600b307413c2af1fb96bb27",
}

ROBOTIQ_2F85_V4_FILES = {
    "robotiq_2f85_v4/2f85.xml": "fb1e45e39484d42154a6a2b8f51d340e0f6cf57c",
    "robotiq_2f85_v4/CHANGELOG.md": "e16f0eea1c7bd8d7c8ffe60179899fda350b8318",
    "robotiq_2f85_v4/LICENSE": "78c0f73570702ddd1fa0a7140d3bb7308ddd9774",
    "robotiq_2f85_v4/README.md": "9885fce59b5aeb80985cfc85373af283ab32032d",
    "robotiq_2f85_v4/assets/base.stl": "777c7fd8a016614c01f59df4173f9937e0e4e2b2",
    "robotiq_2f85_v4/assets/base_coupling.stl": "9fb828a98ca6c96da0ebe78d8dbec97e233e02ee",
    "robotiq_2f85_v4/assets/c-a01-85-open.stl": "49cf16e7d9bb4509f6dda45bab8168c1a8a24753",
    "robotiq_2f85_v4/assets/coupler.stl": "8bcec11ffd80fd15c097c4a28167c4d7bc65c409",
    "robotiq_2f85_v4/assets/driver.stl": "ad6f2c1b2f4a77cdf33b4cf971c3ac47d42f80aa",
    "robotiq_2f85_v4/assets/follower.stl": "7add691448535ab18beb011a79e73bf007c23213",
    "robotiq_2f85_v4/assets/spring_link.stl": "5e4bd903891f8e814718cfc6d626bf158e9de9a7",
    "robotiq_2f85_v4/assets/tongue.stl": "53e69d98396f000c0d2221ab57c435f21ed1c872",
}

MENAGERIE_FILES = {
    **KUKA_IIWA_14_FILES,
    **ROBOTIQ_2F85_V4_FILES,
}


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("utf-8")
    digest = hashlib.sha1()
    digest.update(header)
    digest.update(data)
    return digest.hexdigest()


def raw_menagerie_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_COMMIT}/{path}"


def default_asset_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    return root / "third_party" / "mujoco_menagerie"
