from trading_service import evaluate_provider


def test_ac70_provider_timeout_not_success():
    def timeout():
        raise TimeoutError("provider late")

    result = evaluate_provider(timeout)

    assert result.status == "SAGLAYICI_ZAMAN_ASIMI"
    assert not result.is_success and result.action is None
