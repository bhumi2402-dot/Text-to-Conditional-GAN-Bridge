"""
Comprehensive edge cases and security injection test suite.
"""

import pytest
from src.llm_parser import parse_with_rules, FashionClass
from src.guardrails import sanitize_input, validate_guardrails


def test_extreme_quantities():
    # Exactly boundary values 1 and 16
    res1 = parse_with_rules("1 coat")
    assert res1.success is True
    assert res1.data.count == 1

    res16 = parse_with_rules("sixteen dresses")
    assert res16.success is True
    assert res16.data.count == 16

    # Out of boundary
    res17 = parse_with_rules("17 coats")
    assert res17.success is False
    assert "exceeds" in res17.error_message.lower() or "limit" in res17.error_message.lower()

    res999 = parse_with_rules("999 sneakers")
    assert res999.success is False


def test_malicious_injections():
    # Shell injection pattern
    ok, _, err = sanitize_input("four boots; rm -rf /")
    assert ok is False

    ok2, _, err2 = sanitize_input("eval('import os; os.system()')")
    assert ok2 is False

    # HTML/Script tag injection
    ok3, text3, _ = sanitize_input("<script>alert('hack')</script>two sandals")
    assert "<script>" not in text3


def test_synonym_coverage():
    synonym_tests = [
        ("two hoodies", FashionClass.PULLOVER, 2),
        ("three jumpers", FashionClass.PULLOVER, 3),
        ("four sweaters", FashionClass.PULLOVER, 4),
        ("a pair of kicks", FashionClass.SNEAKER, 2),
        ("two running shoes", FashionClass.SNEAKER, 2),
        ("one flip-flop", FashionClass.SANDAL, 1),
        ("two blazers", FashionClass.COAT, 2),
        ("three jeans", FashionClass.TROUSER, 3),
        ("a couple of handbags", FashionClass.BAG, 2),
        ("two booties", FashionClass.ANKLE_BOOT, 2),
    ]
    for prompt, expected_cls, expected_count in synonym_tests:
        res = parse_with_rules(prompt)
        assert res.success is True, f"Failed on '{prompt}': {res.error_message}"
        assert res.data.class_label == expected_cls
        assert res.data.count == expected_count
