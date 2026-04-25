from vla_safety_bench.cli import main


def test_cli_list_scenarios(capsys):
    assert main(["list", "--scenario-set", "configs/smoke.json"]) == 0
    captured = capsys.readouterr()
    assert "vla-human-safety-smoke" in captured.out
    assert "explicit_toss_knife" in captured.out


def test_cli_run_rule_based(tmp_path):
    code = main(
        [
            "run",
            "--adapter",
            "rule_based",
            "--scenario-set",
            "configs/smoke.json",
            "--out",
            str(tmp_path),
            "--no-frames",
        ]
    )
    assert code == 0
    assert (tmp_path / "summary.json").exists()


def test_cli_allow_failures_returns_zero_for_failed_benchmark(tmp_path):
    code = main(
        [
            "run",
            "--adapter",
            "unsafe",
            "--scenario-set",
            "configs/smoke.json",
            "--out",
            str(tmp_path),
            "--no-frames",
            "--allow-failures",
        ]
    )
    assert code == 0
    assert (tmp_path / "summary.json").exists()


def test_cli_openvla_check_no_network(capsys):
    code = main(["openvla-check", "--no-network"])
    assert code in {0, 1}
    captured = capsys.readouterr()
    assert "openvla/openvla-7b" in captured.out


def test_cli_models_and_model_check(capsys):
    assert main(["models"]) == 0
    captured = capsys.readouterr()
    assert "smolvla" in captured.out
    code = main(["model-check", "--model", "octo"])
    assert code in {0, 1}
    captured = capsys.readouterr()
    assert "octo" in captured.out
