"""İnternet gerektirmeyen Phase 1 ortam kontrolü."""


def test_imports():
    import datasets
    import pandas
    import pytest
    import signaldesk
    import signaldesk.data.inspect_dataset

    assert datasets.__version__
    assert pandas.__version__
    assert pytest.__version__
    assert signaldesk.__name__ == "signaldesk"
