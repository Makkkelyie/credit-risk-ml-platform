import pytest

from crm.ingest import DataQualityError, ingest, validate


def test_clean_data_passes_with_documented_quirks(raw):
    report = validate(raw)
    assert report.errors == []
    assert any("education" in w for w in report.warnings)


@pytest.mark.parametrize("corrupt, message", [
    (lambda d: d.assign(id=1), "duplicate"),
    (lambda d: d.assign(age=12), "ages"),
    (lambda d: d.assign(limit_bal=0), "credit limits"),
    (lambda d: d.assign(default_next_month=2), "binary"),
    (lambda d: d.drop(columns=["pay_amt3"]), "missing columns"),
])
def test_hard_failures_block_the_load(raw, tmp_path, corrupt, message):
    with pytest.raises(DataQualityError, match=message):
        ingest(tmp_path / "wh.duckdb", corrupt(raw))
