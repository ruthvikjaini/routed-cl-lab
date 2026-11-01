"""Frozen API cost schedule and token accounting.

Prices are a frozen, dated public-list-price style schedule (2026-11-01),
in the spirit of the paper's Appendix A.7, which prices every model call
under a frozen schedule that accounts for token caching. The schedule is
deliberately generic -- it is not any one provider's price list -- and it
is frozen so that cost comparisons between conditions are apples-to-apples.

- input tokens:  $3.00 / 1M
- cached input tokens (repeated context prefix): $0.30 / 1M
- output tokens: $15.00 / 1M

Per instance, every condition pays the same base prompt (800 input
tokens: the problem statement) and the same output (400 tokens: the
agent's response). Memory systems add retrieved-context tokens on top,
and the routed conditions add one router call (300 input, 150 output,
of which 200 input tokens are cached cluster summaries).
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_INPUT_PER_M = 3.00
PRICE_CACHED_PER_M = 0.30
PRICE_OUTPUT_PER_M = 15.00
SCHEDULE_DATE = "2026-11-01"

BASE_PROMPT_TOKENS = 800
BASE_OUTPUT_TOKENS = 400

ROUTER_INPUT_TOKENS = 300
ROUTER_CACHED_TOKENS = 200
ROUTER_OUTPUT_TOKENS = 150


@dataclass
class TokenBill:
    """Token usage for one instance, split fresh vs cached input."""

    fresh_input: int = 0
    cached_input: int = 0
    output: int = 0

    def cost_usd(self) -> float:
        return (
            self.fresh_input * PRICE_INPUT_PER_M
            + self.cached_input * PRICE_CACHED_PER_M
            + self.output * PRICE_OUTPUT_PER_M
        ) / 1_000_000

    def total_input(self) -> int:
        return self.fresh_input + self.cached_input


def base_bill() -> TokenBill:
    """The tokens every instance pays regardless of condition."""
    return TokenBill(
        fresh_input=BASE_PROMPT_TOKENS,
        cached_input=0,
        output=BASE_OUTPUT_TOKENS,
    )


def router_bill() -> TokenBill:
    """One router call: problem + cluster summaries in, a decision out."""
    return TokenBill(
        fresh_input=ROUTER_INPUT_TOKENS - ROUTER_CACHED_TOKENS,
        cached_input=ROUTER_CACHED_TOKENS,
        output=ROUTER_OUTPUT_TOKENS,
    )
