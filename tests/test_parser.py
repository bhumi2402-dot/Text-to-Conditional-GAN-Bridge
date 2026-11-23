"""
Unit tests for the Fashion Request Parser (Task U1).
Tests strict schema validation, synonym normalization, count clamping,
out-of-scope refusals, and edge cases.
"""

import pytest
from src.llm_parser import parse_with_rules, FashionClass, ParseResult


def test_standard_query():
    res = parse_with_rules("show me four ankle boots")
    assert res.success is True
    assert res.data is not None
    assert res.data.class_label == FashionClass.ANKLE_BOOT
    assert res.data.count == 4


def test_number_words_and_synonyms():
    # "two warm hoodies" -> Pullover, 2
    res1 = parse_with_rules("generate two warm hoodies")
    assert res1.success is True
    assert res1.data.class_label == FashionClass.PULLOVER
    assert res1.data.count == 2

    # "a pair of sneakers" -> Sneaker, 2
    res2 = parse_with_rules("a pair of sneakers please")
    assert res2.success is True
    assert res2.data.class_label == FashionClass.SNEAKER
    assert res2.data.count == 2

    # "six trainers" -> Sneaker, 6
    res3 = parse_with_rules("create six trainers")
    assert res3.success is True
    assert res3.data.class_label == FashionClass.SNEAKER
    assert res3.data.count == 6


def test_tshirt_and_coat_synonyms():
    res1 = parse_with_rules("three graphic tees")
    assert res1.success is True
    assert res1.data.class_label == FashionClass.TSHIRT_TOP
    assert res1.data.count == 3

    res2 = parse_with_rules("one winter jacket")
    assert res2.success is True
    assert res2.data.class_label == FashionClass.COAT
    assert res2.data.count == 1


def test_default_count():
    res = parse_with_rules("I want a dress")
    assert res.success is True
    assert res.data.class_label == FashionClass.DRESS
    assert res.data.count == 1


def test_out_of_range_count_refusal():
    # Count > 16 should refuse cleanly
    res_high = parse_with_rules("generate 500 shoes")
    assert res_high.success is False
    assert "exceeds maximum limit" in res_high.error_message or "range" in res_high.error_message.lower()

    res_zero = parse_with_rules("generate 0 trousers")
    assert res_zero.success is False
    assert "must be at least 1" in res_zero.error_message


def test_out_of_scope_query_refusal():
    res = parse_with_rules("show me a cute cat")
    assert res.success is False
    assert "cat" in res.error_message.lower() or "not" in res.error_message.lower()


def test_empty_and_whitespace_input():
    res1 = parse_with_rules("")
    assert res1.success is False
    assert "empty" in res1.error_message.lower()

    res2 = parse_with_rules("    ")
    assert res2.success is False
    assert "empty" in res2.error_message.lower()


def test_all_ten_fashion_classes():
    mapping = [
        ("t-shirt", FashionClass.TSHIRT_TOP),
        ("trousers", FashionClass.TROUSER),
        ("pullover", FashionClass.PULLOVER),
        ("dress", FashionClass.DRESS),
        ("coat", FashionClass.COAT),
        ("sandals", FashionClass.SANDAL),
        ("shirt", FashionClass.SHIRT),
        ("sneakers", FashionClass.SNEAKER),
        ("bag", FashionClass.BAG),
        ("ankle boots", FashionClass.ANKLE_BOOT),
    ]
    for prompt_word, expected_cls in mapping:
        res = parse_with_rules(f"show me 1 {prompt_word}")
        assert res.success is True, f"Failed for {prompt_word}: {res.error_message}"
        assert res.data.class_label == expected_cls
