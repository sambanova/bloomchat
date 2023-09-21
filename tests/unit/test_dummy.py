import pytest


@pytest.mark.fast
@pytest.mark.parametrize("toggle_input", [True, False])
def test_dummy_function():
    """
    GIVEN a dummy function
    WHEN the function is called
    THEN the test passes
    """
    assert True
