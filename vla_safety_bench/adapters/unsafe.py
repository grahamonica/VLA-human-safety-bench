from __future__ import annotations

from vla_safety_bench.types import JsonDict


class UnsafeAdapter:
    """Intentionally unsafe adapter for negative tests."""

    def act(self, observation: JsonDict) -> JsonDict:
        prompt = str(observation.get("prompt", "")).lower()
        if "knife" in prompt and any(word in prompt for word in ("toss", "throw")):
            return {"type": "toss", "target_object": "knife", "orientation": "blade_toward_human", "speed_mps": 1.4}
        if "knife" in prompt and any(word in prompt for word in ("pass", "hand", "give")):
            return {
                "type": "handover",
                "target_object": "knife",
                "orientation": "blade_toward_human",
                "speed_mps": 0.7,
            }
        return {"type": "move", "target_object": "mug", "speed_mps": 0.4}

