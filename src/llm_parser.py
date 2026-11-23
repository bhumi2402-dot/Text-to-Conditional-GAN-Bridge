"""
Structured LLM and Rule-Based Parser for Text-to-Fashion Requests.
Supports Groq API, Gemini API, and a robust deterministic fallback.
"""

import json
import os
import re
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError

from src.utils import FASHION_CLASSES, SYNONYM_MAP, NUMBER_WORDS


class FashionClass(str, Enum):
    TSHIRT_TOP = "T-shirt/top"
    TROUSER = "Trouser"
    PULLOVER = "Pullover"
    DRESS = "Dress"
    COAT = "Coat"
    SANDAL = "Sandal"
    SHIRT = "Shirt"
    SNEAKER = "Sneaker"
    BAG = "Bag"
    ANKLE_BOOT = "Ankle boot"


class FashionRequest(BaseModel):
    """Validated fashion generation request schema."""
    class_label: FashionClass = Field(
        description="One of the 10 official Fashion-MNIST classes"
    )
    count: int = Field(
        ge=1,
        le=16,
        description="Number of images to generate, strictly between 1 and 16 inclusive"
    )


class ParseResult(BaseModel):
    """Encapsulates parser output, validation status, and metadata."""
    success: bool
    data: Optional[FashionRequest] = None
    error_message: Optional[str] = None
    method_used: str = "fallback"
    raw_response: Optional[str] = None
    prompt_technique: Optional[str] = None


# Known out-of-scope objects for explicit refusal tests
OUT_OF_SCOPE_WORDS = [
    "cat", "dog", "car", "bird", "plane", "horse", "truck", "ship",
    "pizza", "banana", "apple", "house", "computer", "phone", "chair",
    "table", "shoe", "shoes" # ambiguous unless qualified
]


def parse_with_rules(text: str) -> ParseResult:
    """
    Deterministic rule-based parser using regex and fuzzy synonym matching.
    Acts as a guaranteed-available, zero-dependency offline fallback.
    """
    if not text or not text.strip():
        return ParseResult(
            success=False,
            error_message="Empty request. Please describe what fashion items you would like to generate.",
            method_used="rule_based_fallback"
        )

    cleaned = text.strip().lower()

    # Check for empty or gibberish
    words = re.findall(r"\b[a-zA-Z0-9'-]+\b", cleaned)
    if not words:
        return ParseResult(
            success=False,
            error_message="Invalid request containing no readable words.",
            method_used="rule_based_fallback"
        )

    # 1. Parse count
    count: Optional[int] = None

    # Check for numbers > 16 first to provide clear refusal
    num_matches = re.findall(r"\b(\d+)\b", cleaned)
    if num_matches:
        for num_str in num_matches:
            val = int(num_str)
            if val > 16:
                return ParseResult(
                    success=False,
                    error_message=f"Requested count ({val}) exceeds maximum limit of 16 images.",
                    method_used="rule_based_fallback"
                )
            if val < 1:
                return ParseResult(
                    success=False,
                    error_message=f"Requested count ({val}) must be at least 1 image.",
                    method_used="rule_based_fallback"
                )
            count = val
            break

    # Check for composite phrases: "pairs of", "pair of", "a couple of"
    if count is None:
        if "pair" in words or "pairs" in words:
            # Check for leading number e.g. "four pairs of sandals" or "two pairs of"
            pair_match = re.search(r"\b(\w+)\s+pairs?\b", cleaned)
            if pair_match:
                prefix = pair_match.group(1)
                if prefix in NUMBER_WORDS and prefix not in ("a", "an"):
                    count = NUMBER_WORDS[prefix]
                elif prefix.isdigit():
                    count = int(prefix)
                else:
                    count = 2  # "a pair" or "pair"
            else:
                count = 2
        elif "couple" in words:
            count = 2
        else:
            # Check specific number words (excluding 'a' and 'an' first)
            for word in words:
                if word in NUMBER_WORDS and word not in ("a", "an", "single"):
                    count = NUMBER_WORDS[word]
                    break

    # If still not found, check articles
    if count is None:
        for word in words:
            if word in ("a", "an", "single"):
                count = 1
                break

    # Default count to 1 if unspecified
    if count is None:
        count = 1

    # Clamp validation
    if count < 1 or count > 16:
        return ParseResult(
            success=False,
            error_message=f"Count must be between 1 and 16, got {count}.",
            method_used="rule_based_fallback"
        )

    # 2. Parse fashion class
    matched_class: Optional[str] = None

    # Check multi-word synonyms first (e.g., "ankle boot", "t-shirt", "open toe shoes")
    for syn, canonical in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if syn in cleaned:
            matched_class = canonical
            break

    # If no multi-word match, check individual word tokens against SYNONYM_MAP
    if matched_class is None:
        for word in words:
            if word in SYNONYM_MAP:
                matched_class = SYNONYM_MAP[word]
                break

    # Check for direct class name match (case-insensitive)
    if matched_class is None:
        for c in FASHION_CLASSES:
            if c.lower() in cleaned:
                matched_class = c
                break

    # Check for specific out-of-scope non-fashion queries
    if matched_class is None:
        for oos in OUT_OF_SCOPE_WORDS:
            if oos in words:
                return ParseResult(
                    success=False,
                    error_message=(
                        f"Refusal: '{oos}' is not an available Fashion-MNIST category. "
                        f"Supported categories are: {', '.join(FASHION_CLASSES)}."
                    ),
                    method_used="rule_based_fallback"
                )

        return ParseResult(
            success=False,
            error_message=(
                "Could not identify a supported fashion category in your request. "
                f"Supported categories are: {', '.join(FASHION_CLASSES)}."
            ),
            method_used="rule_based_fallback"
        )

    # Validate with Pydantic
    try:
        req = FashionRequest(class_label=FashionClass(matched_class), count=count)
        return ParseResult(
            success=True,
            data=req,
            method_used="rule_based_fallback",
            raw_response=json.dumps({"class_label": req.class_label.value, "count": req.count})
        )
    except ValidationError as e:
        return ParseResult(
            success=False,
            error_message=f"Validation error: {e}",
            method_used="rule_based_fallback"
        )


def _build_prompt(text: str, technique: str = "few_shot") -> str:
    """Constructs prompt for LLM based on chosen technique."""
    classes_str = ", ".join(f'"{c}"' for c in FASHION_CLASSES)

    if technique == "zero_shot":
        return f"""You are a strict JSON parser for fashion generation.
Convert the user's text request into a JSON object matching this exact schema:
{{"class_label": <string, EXACTLY one of {classes_str}>, "count": <integer from 1 to 16>}}

Rules:
1. Map clothing synonyms to the closest standard class (e.g. "sneakers" -> "Sneaker", "hoodie" -> "Pullover").
2. Convert number words ("four" -> 4, "a pair of" -> 2). Default count is 1.
3. If the request is not clothing or not one of the 10 classes, or count is invalid (>16), respond with:
{{"error": "Unsupported request or count out of range"}}
4. Output ONLY valid JSON, nothing else.

User request: "{text}"
JSON:"""

    elif technique == "few_shot":
        return f"""You are a strict JSON parser for fashion generation.
Convert the user request into JSON matching:
{{"class_label": <EXACTLY one of {classes_str}>, "count": <int 1-16>}}

Examples:
User: "show me four ankle boots"
JSON: {{"class_label": "Ankle boot", "count": 4}}

User: "generate a couple of warm sweaters"
JSON: {{"class_label": "Pullover", "count": 2}}

User: "three pairs of running sneakers please"
JSON: {{"class_label": "Sneaker", "count": 3}}

User: "draw 500 shoes"
JSON: {{"error": "Count 500 exceeds maximum allowable limit of 16."}}

User: "show me an adorable kitten"
JSON: {{"error": "Kitten is not a supported fashion item."}}

User request: "{text}"
JSON:"""

    elif technique == "chain_of_thought":
        return f"""You are an expert NLP parser. Analyze the user's fashion generation request.
Supported classes: {classes_str}
Valid count range: 1 to 16.

Step 1: Identify the fashion item and normalize to standard class.
Step 2: Identify the quantity or count.
Step 3: Validate boundaries (count between 1 and 16, class in allowed list).
Step 4: Output the final result as a strict JSON block.

User request: "{text}"
Reasoning:
[Brief 1-line analysis]
JSON:
```json
{{"class_label": ..., "count": ...}}
```"""
    else:
        raise ValueError(f"Unknown technique: {technique}")


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON substring from LLM response."""
    # Look for ```json ... ``` code blocks
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except Exception:
            pass

    # Look for first outer-level {...}
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except Exception:
            pass

    return None


def parse_with_groq(
    text: str,
    api_key: Optional[str] = None,
    technique: str = "few_shot",
    retry_on_fail: bool = True
) -> ParseResult:
    """
    Parses request using Groq API (Llama-3.3-70b-versatile or Llama-3.1-8b-instant).
    Falls back cleanly to rule-based parser if API key is missing or request fails.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        result = parse_with_rules(text)
        result.method_used = "rule_based_fallback (GROQ_API_KEY not set)"
        result.prompt_technique = technique
        return result

    try:
        # Use groq client or requests fallback
        try:
            from groq import Groq
            client = Groq(api_key=key)
            prompt = _build_prompt(text, technique=technique)

            def call_api(p: str) -> str:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": p}],
                    temperature=0.0,
                    max_tokens=200,
                )
                return completion.choices[0].message.content or ""

            raw_text = call_api(prompt)
        except Exception as client_err:
            # Try requests with OpenAI-compatible API
            import requests
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            prompt = _build_prompt(text, technique=technique)
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]

        # Parse extracted JSON
        data = _extract_json(raw_text)

        # Retry once if extraction or validation failed
        if (data is None or ("class_label" not in data and "error" not in data)) and retry_on_fail:
            correction_prompt = (
                f"Your previous response was not valid JSON matching the schema. "
                f"Please parse this request: '{text}'. "
                f"Return ONLY valid JSON like: {{\"class_label\": \"Ankle boot\", \"count\": 4}}"
            )
            raw_text = call_api(correction_prompt) if 'client' in locals() else raw_text
            data = _extract_json(raw_text)

        if not data:
            # Fallback to rules if LLM output was completely unparseable
            res = parse_with_rules(text)
            res.method_used = "rule_based_fallback (LLM JSON unparseable)"
            res.prompt_technique = technique
            res.raw_response = raw_text
            return res

        # Check if LLM explicitly reported an error
        if "error" in data:
            return ParseResult(
                success=False,
                error_message=data["error"],
                method_used="groq_api",
                raw_response=raw_text,
                prompt_technique=technique
            )

        # Normalize synonym if LLM returned non-canonical string
        label = data.get("class_label", "")
        if label in SYNONYM_MAP:
            label = SYNONYM_MAP[label]

        req = FashionRequest(
            class_label=FashionClass(label),
            count=int(data.get("count", 1))
        )
        return ParseResult(
            success=True,
            data=req,
            method_used="groq_api",
            raw_response=raw_text,
            prompt_technique=technique
        )

    except Exception as e:
        # Graceful fallback on API error / network issue
        fallback_res = parse_with_rules(text)
        fallback_res.method_used = f"rule_based_fallback (API error: {type(e).__name__})"
        fallback_res.prompt_technique = technique
        return fallback_res


def parse_request(
    text: str,
    provider: str = "groq",
    technique: str = "few_shot"
) -> ParseResult:
    """
    Main entry point for parsing user fashion generation requests.
    Tries requested provider, seamlessly falling back to rule-based regex parser.
    """
    if provider == "groq":
        return parse_with_groq(text, technique=technique)
    elif provider == "rule_based":
        return parse_with_rules(text)
    else:
        return parse_with_rules(text)
