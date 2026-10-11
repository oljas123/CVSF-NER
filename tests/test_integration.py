import pytest

pytestmark = [pytest.mark.neural, pytest.mark.integration]


def test_complete_cpu_pipeline(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from cvsf.smoke import run

    result = run(tmp_path / "synthetic")
    assert result["passed"] and result["systems"] == 8
