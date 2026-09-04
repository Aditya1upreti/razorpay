"""
Demo Controls — Live outage simulation for pitch demos

Provides a single-shot flag that, when set, forces the NEXT
ai_investigator.investigate() call to raise an exception on the Gemini API
call, triggering the existing graceful-degradation fallback path (rule-only
scoring, degraded_mode=True).

This reuses the EXACT same code path validated by the existing automated
degraded-mode test — it does not create a separate "fake" degraded path.

Usage:
    from engine.demo_controls import trigger_outage, pop_outage_flag
    trigger_outage()                              # arm the flag
    result = investigate(txn, rule_breakdown)     # triggers fallback
    # flag auto-resets after pop_outage_flag() or after the investigate() call
"""

_force_outage = False


def trigger_outage():
    """Arm the single-shot outage flag. The next _call_gemini() call will
    raise RuntimeError, triggering the existing fallback path."""
    global _force_outage
    _force_outage = True


def pop_outage_flag() -> bool:
    """Return the current flag value and reset it to False (single-shot).
    Called by _call_gemini to check and consume the flag."""
    global _force_outage
    val = _force_outage
    _force_outage = False
    return val
