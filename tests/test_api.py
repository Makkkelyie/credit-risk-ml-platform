import numpy as np
from fastapi.testclient import TestClient

from crm.features import BEHAVIOURAL
from crm.serve import create_app


class DelayModel:
    """Stand-in model: PD rises with last month's delay. Checks the features it receives."""

    def predict_proba(self, X):
        assert set(BEHAVIOURAL) <= set(X.columns)
        p = np.clip(0.1 + 0.3 * X["delay_m1"].to_numpy(), 0, 0.99)
        return np.column_stack([1 - p, p])


ACCOUNT = {
    "limit_bal": 50_000,
    "pay_status": [0, 0, -1, -1, 0, 0],
    "bill_amt": [20_000, 18_000, 15_000, 12_000, 10_000, 9_000],
    "pay_amt": [2_000, 2_000, 1_500, 1_200, 1_000, 900],
}
client = TestClient(create_app(DelayModel(), version="test"))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_low_and_high_risk():
    low = client.post("/score", json=ACCOUNT).json()
    high = client.post("/score", json={**ACCOUNT, "pay_status": [2, 1, 0, 0, 0, 0]}).json()
    assert low == {"probability_of_default": 0.1, "risk_band": "low", "model_version": "test"}
    assert high["probability_of_default"] == 0.7 and high["risk_band"] == "high"


def test_rejects_malformed_history():
    assert client.post("/score", json={**ACCOUNT, "bill_amt": [1, 2, 3]}).status_code == 422
    assert client.post("/score", json={**ACCOUNT, "limit_bal": 0}).status_code == 422
