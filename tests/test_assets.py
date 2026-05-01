from vla_safety_bench.assets import (
    KUKA_IIWA_14_FILES,
    ROBOTIQ_2F85_V4_FILES,
    MENAGERIE_COMMIT,
    MENAGERIE_REPO,
    git_blob_sha1,
    raw_menagerie_url,
)


def test_menagerie_manifest_is_pinned():
    assert MENAGERIE_REPO == "google-deepmind/mujoco_menagerie"
    assert len(MENAGERIE_COMMIT) == 40
    assert "kuka_iiwa_14/scene.xml" in KUKA_IIWA_14_FILES
    assert "robotiq_2f85_v4/2f85.xml" in ROBOTIQ_2F85_V4_FILES
    assert "refs/heads/main" not in raw_menagerie_url("kuka_iiwa_14/scene.xml")


def test_git_blob_sha1_matches_known_empty_blob():
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
