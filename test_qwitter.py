import pytest
import json
from unittest.mock import patch
from qwitter import (
    validate_and_clean_one_word,
    load_api_key_from_env,
    ask_qwitter
)

def test_validate_and_clean_one_word_simple():
    assert validate_and_clean_one_word("Delhi") == "Delhi"
    assert validate_and_clean_one_word("  delhi  ") == "delhi"
    assert validate_and_clean_one_word("ans = Delhi.") == "Delhi"
    assert validate_and_clean_one_word("**Delhi**") == "Delhi"
    assert validate_and_clean_one_word("```json\nDelhi\n```") == "Delhi"

def test_validate_and_clean_one_word_multi_word_fallback():
    # If API returns multiple words despite prompt, Python extracts the most appropriate single word
    assert validate_and_clean_one_word("New Delhi", question="capital of india") == "Delhi"
    assert validate_and_clean_one_word("The Sun") == "Sun"

def test_validate_and_clean_one_word_invalid():
    with pytest.raises(ValueError):
        validate_and_clean_one_word("")
    with pytest.raises(ValueError):
        validate_and_clean_one_word("   ...   ")

def test_load_api_key_from_env():
    key = load_api_key_from_env()
    assert key is not None
    assert key.startswith("AQ.Ab8RN6")

def test_ask_qwitter_empty_question():
    res = ask_qwitter("")
    assert res["status"] == "error"
    assert res["answer"] is None
    assert "empty" in res["error"].lower()

@patch("qwitter.load_api_key_from_env")
def test_ask_qwitter_missing_api_key(mock_env):
    mock_env.return_value = None
    res = ask_qwitter("What is the capital of India?")
    assert res["status"] == "error"
    assert res["answer"] is None
    assert "api key not found" in res["error"].lower()
