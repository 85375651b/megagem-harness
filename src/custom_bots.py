"""Public bot names for the CLI; add imports and __all__ entries for new bots."""

from bot_definitions.fixed_rule_bot import FixedRuleBot
from bot_definitions.sampling_bot import SamplingBot
from bot_definitions.small_bid_bot import SmallBidBot
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot

__all__ = [
    "FixedRuleBot",
    "SmallBidBot",
    "ValuationBot",
    "SoftTrackingBot",
    "SamplingBot",
]
