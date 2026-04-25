from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from vla_safety_bench.types import HumanState, JsonDict, ObjectState


@dataclass(frozen=True)
class HumanTimelineEntry:
    step: int
    humans: list[HumanState]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HumanTimelineEntry":
        return cls(
            step=int(data["step"]),
            humans=[HumanState.from_dict(item) for item in data.get("humans", [])],
        )

    def to_dict(self) -> JsonDict:
        return {"step": self.step, "humans": [human.to_dict() for human in self.humans]}


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    title: str
    category: str
    prompt: str
    max_steps: int
    humans: list[HumanState] = field(default_factory=list)
    human_timeline: list[HumanTimelineEntry] = field(default_factory=list)
    objects: list[ObjectState] = field(default_factory=list)
    expectation: JsonDict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, default_max_steps: int = 3) -> "ScenarioSpec":
        timeline = [HumanTimelineEntry.from_dict(item) for item in data.get("human_timeline", [])]
        timeline.sort(key=lambda item: item.step)
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", data["id"])),
            category=str(data.get("category", "uncategorized")),
            prompt=str(data["prompt"]),
            max_steps=int(data.get("max_steps", default_max_steps)),
            humans=[HumanState.from_dict(item) for item in data.get("humans", [])],
            human_timeline=timeline,
            objects=[ObjectState.from_dict(item) for item in data.get("objects", [])],
            expectation=dict(data.get("expectation", {})),
            tags=[str(item) for item in data.get("tags", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def humans_at(self, step_index: int) -> list[HumanState]:
        if not self.human_timeline:
            return list(self.humans)
        selected: list[HumanState] | None = None
        for entry in self.human_timeline:
            if entry.step <= step_index:
                selected = entry.humans
            else:
                break
        return list(selected if selected is not None else self.humans)

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "prompt": self.prompt,
            "max_steps": self.max_steps,
            "humans": [human.to_dict() for human in self.humans],
            "human_timeline": [entry.to_dict() for entry in self.human_timeline],
            "objects": [obj.to_dict() for obj in self.objects],
            "expectation": self.expectation,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ScenarioSet:
    name: str
    description: str
    scenarios: list[ScenarioSpec]
    default_max_steps: int = 3

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "default_max_steps": self.default_max_steps,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def load_scenario_set(path: str | Path) -> ScenarioSet:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    default_max_steps = int(payload.get("default_max_steps", 3))
    scenarios = [
        ScenarioSpec.from_dict(item, default_max_steps=default_max_steps)
        for item in payload.get("scenarios", [])
    ]
    if not scenarios:
        raise ValueError(f"No scenarios found in {scenario_path}")
    return ScenarioSet(
        name=str(payload.get("name", scenario_path.stem)),
        description=str(payload.get("description", "")),
        default_max_steps=default_max_steps,
        scenarios=scenarios,
    )

