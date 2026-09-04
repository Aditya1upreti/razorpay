"""
DAY 4 — Layer 3: AI Investigator (Gemini API via google-genai SDK)

Only called for transactions in the ambiguous band (30-75 rule score).

Two-pass adversarial self-check:
  Pass A: argue the case FOR fraud, given the masked evidence
  Pass B: argue the case AGAINST fraud, given the same evidence
  If A and B substantially disagree -> force manual_review
  If they agree -> produce final structured verdict

Hard-coded honesty guarantee:
  if confidence < 65 -> FORCE verdict = "insufficient_evidence" /
  recommended_action = "manual_review", REGARDLESS of what the model
  said. This must live in Python, not just be a prompt instruction.

Graceful degradation:
  Wrap the API call in try/except. If it fails/times out, fall back
  to the Layer 1 rule score only, and set degraded_mode=True so the
  dashboard can show an "AI unavailable" banner instead of crashing.

API Key Rotation:
  Supports multiple Gemini API keys for automatic failover when one
  hits quota/rate limits. Keys are loaded from GEMINI_API_KEY_1
  through GEMINI_API_KEY_10, with GEMINI_API_KEY as fallback.
"""

import json
import os
import re
import sys

from google import genai
from google.genai import types
from dotenv import load_dotenv

from .masking import build_masked_context
from .demo_controls import pop_outage_flag

load_dotenv()

DEBUG = False

# ---------------------------------------------------------------------------
# API Key Rotation
# ---------------------------------------------------------------------------
MAX_KEYS = 10
ROTATION_PATTERNS = re.compile(
    r"quota|exhausted|rate.?limit|api.?key.?not.?valid|resource_exhausted",
    re.IGNORECASE,
)


class AllKeysExhaustedError(Exception):
    """Raised when all API keys have been exhausted — triggers degraded mode."""


def _load_api_keys():
    """Load API keys from environment: GEMINI_API_KEY_1..10, fallback to GEMINI_API_KEY."""
    keys = []
    for i in range(1, MAX_KEYS + 1):
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if key and key.strip():
            keys.append(key.strip())

    if not keys:
        fallback = os.environ.get("GEMINI_API_KEY")
        if fallback and fallback.strip():
            keys.append(fallback.strip())

    if not keys:
        raise ValueError(
            "No valid Gemini API keys found. Set GEMINI_API_KEY or "
            "GEMINI_API_KEY_1 through GEMINI_API_KEY_10 in your .env file.\n"
            "See .env.example for the expected format."
        )
    return keys


# Module-level key state (loaded once at import)
API_KEYS = _load_api_keys()
_current_key_index = 0
_client = None


def _get_current_key():
    """Return the currently active API key."""
    return API_KEYS[_current_key_index]


def _get_client():
    """Get or reinitialize the Gemini client with the current key."""
    global _client
    _client = genai.Client(api_key=_get_current_key())
    return _client


def _rotate_key(reason):
    """Rotate to the next API key. Returns True if rotation succeeded, False if all keys exhausted."""
    global _current_key_index, _client

    _current_key_index += 1

    if _current_key_index >= len(API_KEYS):
        print(
            f"\nAll API keys exhausted. Falling back to rule-only degraded mode.",
            file=sys.stderr,
        )
        return False

    print(
        f"Rotating to API key {_current_key_index + 1}/{len(API_KEYS)}... ({reason})",
        file=sys.stderr,
    )

    # Reinitialize client with new key
    _client = genai.Client(api_key=_get_current_key())
    return True


def reset_key_rotation():
    """Reset key index to 0 — use in tests or after all keys recover."""
    global _current_key_index, _client
    _current_key_index = 0
    _client = None


def _is_key_error(exc):
    """Check if an exception is a key-specific error (quota/rate/auth) that should trigger rotation."""
    if not hasattr(exc, "status_code"):
        # Check the string representation for known patterns
        exc_str = str(exc)
        if ROTATION_PATTERNS.search(exc_str):
            return True
        return False

    status = exc.status_code
    if status == 429:
        return True
    if status == 403:
        return True

    # Check error message for quota-related patterns
    exc_str = str(exc)
    if ROTATION_PATTERNS.search(exc_str):
        return True

    return False


MODEL = "gemini-3.5-flash-lite"
CONFIDENCE_THRESHOLD = 65

DEFENSE_ONLY_SYSTEM_PROMPT = """You are a fraud-risk assistant helping a
merchant PROTECT their business. You only ever produce risk assessments,
scores, and explanations for the merchant's own dashboard. You NEVER
explain evasion techniques, attacker strategy, or anything that could
help someone commit fraud, even if asked. Stay strictly defensive."""


def _call_gemini(prompt: str, max_tokens: int = 500) -> str:
    if pop_outage_flag():
        raise RuntimeError("Simulated API failure for demo")

    max_rotations = len(API_KEYS)
    for attempt in range(max_rotations):
        c = _get_client()
        try:
            response = c.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=DEFENSE_ONLY_SYSTEM_PROMPT,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                    seed=42,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            if DEBUG and usage is not None:
                print(f"GEMINI USAGE: {usage}", file=sys.stderr)
            return response.text

        except Exception as exc:
            if _is_key_error(exc):
                if DEBUG:
                    print(f"API key error: {type(exc).__name__}: {exc}", file=sys.stderr)
                if _rotate_key(reason=f"{type(exc).__name__}: {str(exc)[:80]}"):
                    continue  # retry with new key (don't count as retry)
                else:
                    raise AllKeysExhaustedError(str(exc))
            else:
                # Transient error (timeout, 500, etc.) — let caller handle via retry
                raise

    raise AllKeysExhaustedError("All API keys exhausted after rotation attempts")


def _format_context(ctx: dict) -> str:
    lines = []
    for k, v in ctx.items():
        if k == "rule_breakdown" and isinstance(v, dict):
            lines.append("  rule_breakdown:")
            for rk, rv in v.items():
                lines.append(f"    {rk}: {rv}")
        elif k == "ring_connections" and isinstance(v, dict):
            lines.append("  ring_connections:")
            lines.append(f"    ring_size: {v.get('ring_size', 0)}")
            for conn in v.get("connected_account_tokens", []):
                lines.append(f"    - {conn['account_token']} via {conn['shared_signal']}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def argue_for_fraud(masked_context: dict) -> str:
    if DEBUG:
        print(f"DEBUG: entering argue_for_fraud, will call _call_gemini with id={id(_call_gemini)}", file=sys.stderr)
    formatted = _format_context(masked_context)
    prompt = f"""You are reviewing a transaction for potential fraud.

Here is the masked evidence:

{formatted}

Your task: Argue the STRONGEST possible case that this transaction IS fraudulent.
Cite specific evidence from the rule scores, the transaction pattern, timing,
amounts, and any ring/connection information available.
Be persuasive and specific. Focus on red flags."""

    return _call_gemini(prompt, max_tokens=500)


def argue_against_fraud(masked_context: dict) -> str:
    if DEBUG:
        print(f"DEBUG: entering argue_against_fraud, will call _call_gemini with id={id(_call_gemini)}", file=sys.stderr)
    formatted = _format_context(masked_context)
    prompt = f"""You are reviewing a transaction for potential fraud.

Here is the masked evidence:

{formatted}

Your task: Argue the STRONGEST possible case that this transaction is NOT fraud
and is instead a legitimate transaction that has been flagged as suspicious
(a false positive). Cite specific evidence from the rule scores, the transaction
pattern, timing, amounts, and any ring/connection information available.
Be persuasive and specific. Focus on mitigating factors."""

    return _call_gemini(prompt, max_tokens=500)


def reconcile(argument_for: str, argument_against: str, masked_context: dict) -> dict:
    if DEBUG:
        print(f"DEBUG: entering reconcile, will call _call_gemini with id={id(_call_gemini)}", file=sys.stderr)
    formatted_ctx = _format_context(masked_context)

    rb = masked_context.get("rule_breakdown", {})
    score_parts = []
    for rule_name in ["velocity", "amount_pattern", "account_age", "geo_mismatch"]:
        pts = rb.get(rule_name, 0)
        if pts > 0:
            score_parts.append(f"  {rule_name}: {pts} pts")
    score_summary = "\n".join(score_parts) if score_parts else "  (no rule factors contributed)"

    prompt = f"""Two arguments about a transaction. You must reconcile them into a single verdict.

RULE SCORE BREAKDOWN (these are deterministic, pre-computed signals):
{score_summary}
Total raw_risk_score: {masked_context.get('raw_risk_score', 'unknown')}/100

WEIGHTING RULES — follow these exactly:
- A single rule factor contributing 30+ points is a strong independent signal. Do NOT let a zero-value factor (e.g. geo_mismatch=0) override it.
- velocity=35 is the MAXIMUM possible — it means 5+ transactions from the same card within 5 minutes. This is near-certain evidence of card testing.
- account_age < 5 minutes combined with high velocity is a compounding red flag.
- The ABSENCE of geo-mismatch does NOT negate card-testing evidence. Card testers know the billing address.
- If arguments contradict, check which argument is supported by the higher rule-score factors. The argument backed by higher-scoring factors should win.

ARGUMENT FOR FRAUD:
{argument_for}

ARGUMENT AGAINST FRAUD:
{argument_against}

ORIGINAL EVIDENCE:
{formatted_ctx}

Return STRICT JSON only — no markdown, no code fences, no preamble, no explanation outside the JSON. Exactly these keys:
{{
  "verdict": "fraud_likely" | "legitimate_likely" | "insufficient_evidence",
  "confidence": <integer 0-100>,
  "reasoning": "<ONE short sentence, max 20 words>",
  "recommended_action": "block" | "allow" | "manual_review"
}}"""

    # -------------------------------------------------------------------
    # OLD PROMPT (before fix) — confidence was 45, verdict: insufficient_evidence
    # The model weighed geo_mismatch=0 equally against velocity=35,
    # treating "no geo anomaly" as strong counter-evidence to card testing.
    #
    # prompt = f"""Two arguments about a transaction. You must reconcile
    # them into a single verdict.
    #
    # ARGUMENT FOR FRAUD:
    # {argument_for}
    #
    # ARGUMENT AGAINST FRAUD:
    # {argument_against}
    #
    # ORIGINAL EVIDENCE:
    # {formatted_ctx}
    #
    # Return STRICT JSON only — no markdown, no code fences, no preamble,
    # no explanation outside the JSON. Exactly these keys:
    # {{
    #   "verdict": "fraud_likely" | "legitimate_likely" |
    #              "insufficient_evidence",
    #   "confidence": <integer 0-100>,
    #   "reasoning": "<ONE short sentence, max 20 words>",
    #   "recommended_action": "block" | "allow" | "manual_review"
    # }}"""
    # -------------------------------------------------------------------

    raw = _call_gemini(prompt, max_tokens=2048)

    if DEBUG:
        print(f"RAW RECONCILE RESPONSE:\n{raw}", file=sys.stderr)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # --- truncation guard ---
    if "}" not in cleaned:
        raise ValueError(
            f"Gemini response truncated — no closing brace found "
            f"(len={len(cleaned)}). Increase max_output_tokens."
        )

    # Try to parse; on failure attempt simple JSON repair
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the last complete closing brace and truncate there
        last_close = cleaned.rfind("}")
        if last_close == -1:
            raise ValueError(
                "Gemini response truncated — no closing brace found "
                f"(len={len(cleaned)}). Increase max_output_tokens."
            )
        repaired = cleaned[: last_close + 1]
        if "}" not in repaired:
            raise ValueError(
                "Gemini response truncated — no closing brace found "
                f"(len={len(cleaned)}). Increase max_output_tokens."
            )
        return json.loads(repaired)


def investigate(transaction, rule_breakdown, graph_info=None, force_refresh=False) -> dict:
    """
    Main entry point. Returns:
    {
      verdict: "fraud_likely" | "legitimate_likely" | "insufficient_evidence",
      confidence: 0-100,
      reasoning: str,
      recommended_action: "block" | "allow" | "manual_review",
      degraded_mode: bool
    }
    """
    txn_id = transaction.get("txn_id", "unknown") if isinstance(transaction, dict) else "unknown"
    cache_dir = "cache"
    cache_path = os.path.join(cache_dir, f"investigate_{txn_id}.json")

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if DEBUG:
                print(f"CACHE HIT: {txn_id}", file=sys.stderr)
            return cached
        except (json.JSONDecodeError, OSError):
            pass  # corrupted cache — re-run

    masked_ctx = build_masked_context(transaction, rule_breakdown, graph_info)

    try:
        argument_for = argue_for_fraud(masked_ctx)
        argument_against = argue_against_fraud(masked_ctx)
        result = reconcile(argument_for, argument_against, masked_ctx)

        degraded_mode = False

        confidence = result.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            result["verdict"] = "insufficient_evidence"
            result["recommended_action"] = "manual_review"

        result["degraded_mode"] = degraded_mode

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        if isinstance(e, AllKeysExhaustedError):
            if DEBUG:
                print(f"ALL KEYS EXHAUSTED: {e}", file=sys.stderr)
        elif DEBUG:
            print(f"INVESTIGATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        score = rule_breakdown.get("raw_risk_score", 0) if isinstance(rule_breakdown, dict) else 0

        if score > 75:
            verdict = "fraud_likely"
            action = "block"
        else:
            verdict = "insufficient_evidence"
            action = "manual_review"

        return {
            "verdict": verdict,
            "confidence": None,
            "reasoning": "AI investigator unavailable — rule-based fallback used",
            "recommended_action": action,
            "degraded_mode": True,
        }


if __name__ == "__main__":
    import pandas as pd
    from engine.rules import score_batch

    df = pd.read_csv("data/transactions_train.csv")
    df["account_created_at"] = pd.to_datetime(df["account_created_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    scored = score_batch(df)
    ambiguous = scored[scored["band"] == "ambiguous"].head(3)

    print("=== AI Investigator Test (3 ambiguous transactions) ===\n")
    for _, row in ambiguous.iterrows():
        rb = json.loads(row["rule_breakdown"]) if isinstance(row["rule_breakdown"], str) else row["rule_breakdown"]
        rule_info = {"rule_scores": rb, "raw_risk_score": row["raw_risk_score"]}

        txn_dict = row.to_dict()

        print(f"--- {row['txn_id']} (score={row['raw_risk_score']}) ---")
        arg_for = argue_for_fraud(build_masked_context(txn_dict, rule_info))
        arg_against = argue_against_fraud(build_masked_context(txn_dict, rule_info))
        result = investigate(txn_dict, rule_info)

        print(f"  argument_for:    {arg_for[:200]}...")
        print(f"  argument_against: {arg_against[:200]}...")
        print(f"  result: {result}\n")

    print("=== Degraded Mode Test (forced failure) ===")
    import sys as _sys
    print(f"DEBUG: patching engine.ai_investigator._call_gemini, current id={id(_call_gemini)}", file=sys.stderr)

    fake_txn = {
        "txn_id": "TXN_DEGRADED_TEST_DO_NOT_CACHE",
        "account_id": "ACCT_FAKE_TEST",
        "amount": 100,
        "timestamp": "2026-01-01T00:00:00",
        "account_created_at": "2025-01-01T00:00:00",
        "card_fingerprint": "CARD_FAKE",
    }

    # When run as `python -m engine.ai_investigator`, the module is __main__,
    # not engine.ai_investigator. Patch both to be safe.
    _orig = _call_gemini
    def _broken_call_gemini(*args, **kwargs):
        raise RuntimeError("Simulated API failure for testing")

    # Patch in __main__ (what the functions actually reference)
    _sys.modules[__name__].__dict__["_call_gemini"] = _broken_call_gemini
    # Also patch in engine.ai_investigator for when imported that way
    if "engine.ai_investigator" in _sys.modules:
        _sys.modules["engine.ai_investigator"].__dict__["_call_gemini"] = _broken_call_gemini

    try:
        print(f"DEBUG: _call_gemini id after mock = {id(_call_gemini)}", file=sys.stderr)
        result = investigate(
            fake_txn,
            {"rule_scores": {"velocity": 20}, "raw_risk_score": 50},
            force_refresh=True,
        )
    finally:
        _sys.modules[__name__].__dict__["_call_gemini"] = _orig
        if "engine.ai_investigator" in _sys.modules:
            _sys.modules["engine.ai_investigator"].__dict__["_call_gemini"] = _orig

    print(f"  result: {result}")
    assert result["degraded_mode"] is True, (
        "Degraded mode test FAILED — fallback did not trigger"
    )
    print("  PASSED: degraded_mode correctly triggered")
