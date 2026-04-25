from __future__ import annotations

from vla_safety_bench.types import (
    JsonDict,
    any_human_in_danger_zone,
    prompt_contains_dangerous_fragment,
)


class RuleBasedBaselineAdapter:
    """Transparent baseline used to validate benchmark wiring."""

    def act(self, observation: JsonDict) -> JsonDict:
        if any_human_in_danger_zone(observation):
            return {"type": "stop", "reason": "human in danger zone"}
        prompt = str(observation.get("prompt", "")).lower()
        if prompt_contains_dangerous_fragment(prompt):
            return {"type": "refuse", "reason": "dangerous human-directed instruction"}
        if "knife" in prompt and any(word in prompt for word in ("pass", "hand", "give")):
            return {
                "type": "handover",
                "target_object": "knife",
                "orientation": "handle_toward_human",
                "speed_mps": 0.12,
                "reason": "safe knife handover orientation",
            }
        if "mug" in prompt:
            return {"type": "move", "target_object": "mug", "speed_mps": 0.18}
        return {"type": "wait", "reason": "no recognized task object"}
