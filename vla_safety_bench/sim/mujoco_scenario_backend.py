from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any, Mapping, Sequence

from vla_safety_bench.scenarios import ScenarioSpec
from vla_safety_bench.sim.kinematic_backend import KinematicSimulation
from vla_safety_bench.sim.mujoco_backend import (
    KUKA_DEFAULT_PHYSICS_SUBSTEPS,
    KUKA_JOINT_NAMES,
    apply_joint_controls,
    apply_joint_positions,
    body_position,
    kuka_render_joint_positions,
    kuka_scenario_mujoco_xml,
    read_joint_controls,
    read_joint_positions,
    render_scene_data_png,
    render_scene_png,
    scenario_mujoco_xml,
    set_freejoint_pose,
    set_mocap_body_pose,
    site_position,
)
from vla_safety_bench.sim.mesh_assets import MeshAssetLibrary, load_mesh_asset_library
from vla_safety_bench.types import JsonDict, ObjectState, Observation, RobotAction
from vla_safety_bench.video import camera_frame_path


def _dedupe_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for camera in cameras:
        camera_name = str(camera).strip()
        if camera_name and camera_name not in seen:
            ordered.append(camera_name)
            seen.add(camera_name)
    return tuple(ordered)


class MujocoScenarioSimulation(KinematicSimulation):
    """MuJoCo-rendered scenario backend, with persistent KUKA physics when requested."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        output_dir: Path,
        *,
        render_frames: bool = True,
        backend_name: str = "mujoco-scenario+kinematic",
        camera: str = "bench_cam",
        use_kuka: bool = False,
        mesh_assets: MeshAssetLibrary | str | Path | None = None,
        video_cameras: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            scenario,
            output_dir,
            render_frames=render_frames,
            backend_name=backend_name,
        )
        self.camera = camera
        self.video_cameras = _dedupe_cameras(video_cameras if video_cameras is not None else [camera])
        self.use_kuka = use_kuka
        self.mesh_assets = load_mesh_asset_library(mesh_assets)
        self._mujoco_model: Any | None = None
        self._mujoco_data: Any | None = None
        self._mujoco_substeps = KUKA_DEFAULT_PHYSICS_SUBSTEPS
        self._current_step_index = 0
        self._last_contact_events: list[JsonDict] = []
        self._last_joint_targets: dict[str, float] = {}
        self._robot_body_names: set[str] = set()
        self._physics_human_ids = self._all_scenario_human_ids()
        if self.use_kuka:
            self._initialize_kuka_physics()

    def observe(self, step_index: int) -> Observation:
        self.world.humans = self.scenario.humans_at(step_index)
        self._safety_context_cache = None
        self._current_step_index = step_index
        if self.use_kuka:
            self._sync_humans_to_mujoco()
            self._forward_mujoco()
            self._refresh_world_from_mujoco()
        image_path = None
        camera_frames: dict[str, str] = {}
        if self.render_frames:
            image_path, camera_frames = self._render_mujoco_frames(step_index)
        return Observation(
            scenario_id=self.scenario.id,
            prompt=self.scenario.prompt,
            step_index=step_index,
            humans=list(self.world.humans),
            objects=list(self.world.objects.values()),
            image_path=image_path,
            metadata={
                "backend": self.backend_name,
                "category": self.scenario.category,
                "tags": self.scenario.tags,
                "mujoco_render": image_path is not None,
                "camera": self.camera,
                "camera_frames": camera_frames,
                "robot_model": "kuka_iiwa_14_menagerie" if self.use_kuka else "primitive_proxy",
                "mesh_assets": str(self.mesh_assets.manifest_path) if self.mesh_assets else None,
                "mujoco_physics": self.use_kuka,
                "mujoco_time_s": self._mujoco_time_s() if self.use_kuka else None,
                "kuka_joint_positions": self._kuka_joint_positions() if self.use_kuka else None,
                "kuka_joint_targets": self._kuka_joint_targets() if self.use_kuka else None,
                "world_state": self.world.to_dict(),
                "safety_context": self._safety_context(),
            },
        )

    def apply_action(self, action: RobotAction) -> JsonDict:
        if not self.use_kuka:
            return super().apply_action(action)

        joint_targets = self._joint_targets_from_action(action)
        if joint_targets is not None and not action.is_safety_action:
            events = self._apply_joint_command_action(action, joint_targets)
        else:
            events = self._apply_action_core(action)
            self._sync_objects_to_mujoco()

        events.extend(self._step_kuka_physics(action, joint_targets))
        events.extend(self._proximity_events())
        self.world.events.extend(events)
        return {
            "backend": self.backend_name,
            "events": events,
            "world_state": self.world.to_dict(),
            "mujoco": self._mujoco_feedback(),
        }

    def _proximity_events(self) -> list[JsonDict]:
        events = super()._proximity_events()
        if self.use_kuka and self._last_contact_events:
            events.extend(self._last_contact_events)
            self._last_contact_events = []
        return events

    def _render_mujoco_frames(self, step_index: int) -> tuple[str | None, dict[str, str]]:
        import mujoco

        legacy_frame_path = self.output_dir / "frames" / self.scenario.id / f"{step_index:03d}.png"
        camera_frames: dict[str, str] = {}
        if self.use_kuka:
            if self._mujoco_model is None or self._mujoco_data is None:
                raise RuntimeError("KUKA MuJoCo physics model is not initialized.")
            image_path = render_scene_data_png(
                self._mujoco_model,
                self._mujoco_data,
                legacy_frame_path,
                camera=self.camera,
            )
            for camera in self.video_cameras:
                frame_path = camera_frame_path(self.output_dir, self.scenario.id, camera, step_index)
                if camera == self.camera:
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(image_path, frame_path)
                else:
                    render_scene_data_png(
                        self._mujoco_model,
                        self._mujoco_data,
                        frame_path,
                        camera=camera,
                    )
                camera_frames[camera] = str(frame_path)
            return image_path, camera_frames

        xml = (
            scenario_mujoco_xml(
                objects=list(self.world.objects.values()),
                humans=list(self.world.humans),
                mesh_assets=self.mesh_assets,
            )
        )
        model = mujoco.MjModel.from_xml_string(xml)
        image_path = render_scene_png(
            model,
            legacy_frame_path,
            camera=self.camera,
            joint_positions=None,
        )
        for camera in self.video_cameras:
            frame_path = camera_frame_path(self.output_dir, self.scenario.id, camera, step_index)
            if camera == self.camera:
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                copyfile(image_path, frame_path)
            else:
                render_scene_png(
                    model,
                    frame_path,
                    camera=camera,
                    joint_positions=None,
                )
            camera_frames[camera] = str(frame_path)
        return image_path, camera_frames

    def _initialize_kuka_physics(self) -> None:
        import mujoco

        xml = kuka_scenario_mujoco_xml(
            objects=list(self.world.objects.values()),
            humans=self._initial_physics_humans(),
            mesh_assets=self.mesh_assets,
        )
        self._mujoco_model = mujoco.MjModel.from_xml_string(xml)
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        self._refresh_robot_body_names()
        initial_targets = kuka_render_joint_positions(0, self.scenario.max_steps)
        apply_joint_positions(self._mujoco_model, self._mujoco_data, initial_targets)
        self._last_joint_targets = apply_joint_controls(
            self._mujoco_model,
            self._mujoco_data,
            initial_targets,
        )
        self._sync_humans_to_mujoco()
        self._forward_mujoco()
        self._refresh_world_from_mujoco()

    def _apply_joint_command_action(
        self,
        action: RobotAction,
        joint_targets: Mapping[str, float],
    ) -> list[JsonDict]:
        events = self._pre_action_safety_events(action)
        self.world.robot.mode = "moving"
        if action.speed_mps is not None:
            self.world.robot.speed_scale = min(max(action.speed_mps / 0.4, 0.0), 1.0)
        else:
            self.world.robot.speed_scale = 1.0
        events.append(
            {
                "type": "state_update",
                "code": "joint_command_received",
                "joint_targets": {name: float(value) for name, value in joint_targets.items()},
            }
        )
        return events

    def _step_kuka_physics(
        self,
        action: RobotAction,
        joint_targets: Mapping[str, float] | None,
    ) -> list[JsonDict]:
        import mujoco

        if self._mujoco_model is None or self._mujoco_data is None:
            raise RuntimeError("KUKA MuJoCo physics model is not initialized.")

        self._sync_humans_to_mujoco()
        applied_targets: dict[str, float] = {}
        if action.type in {"refuse", "stop", "wait"}:
            applied_targets = self._hold_current_kuka_pose()
        elif action.type == "slow":
            if joint_targets is not None:
                applied_targets = self._slow_kuka_targets(joint_targets)
            else:
                applied_targets = self._hold_current_kuka_pose()
        elif joint_targets is not None:
            applied_targets = apply_joint_controls(self._mujoco_model, self._mujoco_data, joint_targets)
        else:
            applied_targets = self._default_kuka_targets_for_step()

        contact_events: list[JsonDict] = []
        seen_contacts: set[tuple[str, str, str]] = set()
        for _ in range(self._mujoco_substeps):
            mujoco.mj_step(self._mujoco_model, self._mujoco_data)
            contact_events.extend(self._collect_contact_events(seen_contacts))

        self._last_joint_targets = applied_targets
        self._last_contact_events = contact_events
        self._refresh_world_from_mujoco()
        return [
            {
                "type": "state_update",
                "code": "mujoco_physics_stepped",
                "time_s": self._mujoco_time_s(),
                "substeps": self._mujoco_substeps,
                "joint_targets": applied_targets,
                "joint_positions": self._kuka_joint_positions(),
            }
        ]

    def _joint_targets_from_action(self, action: RobotAction) -> dict[str, float] | None:
        raw = action.raw
        absolute = self._first_raw_value(
            raw,
            (
                "joint_positions",
                "joint_targets",
                "joint_commands",
                "kuka_joint_positions",
                "kuka_joint_targets",
                "mujoco_joint_targets",
                "qpos",
            ),
        )
        if absolute is not None:
            return self._coerce_joint_mapping(absolute)

        deltas = self._first_raw_value(
            raw,
            (
                "joint_deltas",
                "joint_delta",
                "kuka_joint_deltas",
                "mujoco_joint_deltas",
                "delta_qpos",
            ),
        )
        if deltas is None:
            return None
        delta_map = self._coerce_joint_mapping(deltas)
        current = self._kuka_joint_positions()
        return {name: current.get(name, 0.0) + delta for name, delta in delta_map.items()}

    def _first_raw_value(self, raw: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
        for key in keys:
            if key in raw:
                return raw[key]
        nested = raw.get("raw")
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
        return None

    def _coerce_joint_mapping(self, value: Any) -> dict[str, float]:
        if isinstance(value, Mapping):
            targets: dict[str, float] = {}
            for raw_name, raw_value in value.items():
                joint_name = self._normalize_joint_name(str(raw_name))
                if joint_name is None:
                    raise RuntimeError(f"Unknown KUKA joint command name {raw_name!r}.")
                targets[joint_name] = float(raw_value)
            return targets
        if isinstance(value, (list, tuple)):
            if len(value) != len(KUKA_JOINT_NAMES):
                raise RuntimeError(
                    f"KUKA joint command lists must have {len(KUKA_JOINT_NAMES)} values, got {len(value)}."
                )
            return {name: float(raw_value) for name, raw_value in zip(KUKA_JOINT_NAMES, value, strict=True)}
        raise RuntimeError(f"KUKA joint command must be a mapping or 7-value list, got {value!r}.")

    def _normalize_joint_name(self, name: str) -> str | None:
        compact = name.strip().lower().replace("_", "").replace("-", "")
        for joint_name in KUKA_JOINT_NAMES:
            if compact == joint_name:
                return joint_name
        if compact.startswith("joint") and compact[5:].isdigit():
            candidate = f"joint{int(compact[5:])}"
            if candidate in KUKA_JOINT_NAMES:
                return candidate
        if compact.startswith("j") and compact[1:].isdigit():
            candidate = f"joint{int(compact[1:])}"
            if candidate in KUKA_JOINT_NAMES:
                return candidate
        return None

    def _default_kuka_targets_for_step(self) -> dict[str, float]:
        if self._mujoco_model is None or self._mujoco_data is None:
            raise RuntimeError("KUKA MuJoCo physics model is not initialized.")
        targets = kuka_render_joint_positions(self._current_step_index, self.scenario.max_steps)
        return apply_joint_controls(self._mujoco_model, self._mujoco_data, targets)

    def _hold_current_kuka_pose(self) -> dict[str, float]:
        if self._mujoco_model is None or self._mujoco_data is None:
            raise RuntimeError("KUKA MuJoCo physics model is not initialized.")
        targets = self._kuka_joint_positions()
        self._mujoco_data.qvel[:] = 0.0
        return apply_joint_controls(self._mujoco_model, self._mujoco_data, targets)

    def _slow_kuka_targets(self, joint_targets: Mapping[str, float]) -> dict[str, float]:
        if self._mujoco_model is None or self._mujoco_data is None:
            raise RuntimeError("KUKA MuJoCo physics model is not initialized.")
        current = self._kuka_joint_positions()
        scaled = {
            name: current.get(name, 0.0) + 0.25 * (float(target) - current.get(name, 0.0))
            for name, target in joint_targets.items()
        }
        return apply_joint_controls(self._mujoco_model, self._mujoco_data, scaled)

    def _sync_humans_to_mujoco(self) -> None:
        if self._mujoco_model is None or self._mujoco_data is None:
            return
        current = {human.id: human for human in self.world.humans}
        for index, human_id in enumerate(self._physics_human_ids):
            human = current.get(human_id)
            if human is None:
                position = (10.0 + index, 10.0, 0.45)
            else:
                position = (human.position_m[0], human.position_m[1], 0.45)
            set_mocap_body_pose(self._mujoco_model, self._mujoco_data, human_id, position)

    def _sync_objects_to_mujoco(self) -> None:
        if self._mujoco_model is None or self._mujoco_data is None:
            return
        for obj in self.world.objects.values():
            x, y, z = obj.position_m
            set_freejoint_pose(self._mujoco_model, self._mujoco_data, f"{obj.name}_freejoint", (x, y, z))
        self._forward_mujoco()

    def _refresh_world_from_mujoco(self) -> None:
        if self._mujoco_model is None or self._mujoco_data is None:
            return
        end_effector = site_position(self._mujoco_model, self._mujoco_data, "attachment_site")
        if end_effector is not None:
            self.world.robot.end_effector_m = end_effector
        for name, obj in list(self.world.objects.items()):
            position = body_position(self._mujoco_model, self._mujoco_data, name)
            if position is None:
                continue
            self.world.objects[name] = ObjectState(
                name=obj.name,
                position_m=position,
                orientation=obj.orientation,
                metadata=obj.metadata,
            )

    def _collect_contact_events(self, seen: set[tuple[str, str, str]]) -> list[JsonDict]:
        import mujoco

        if self._mujoco_model is None or self._mujoco_data is None:
            return []
        events: list[JsonDict] = []
        for index in range(self._mujoco_data.ncon):
            contact = self._mujoco_data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_name = mujoco.mj_id2name(self._mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
            geom2_name = mujoco.mj_id2name(self._mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
            body1_name = self._body_name_for_geom(geom1)
            body2_name = self._body_name_for_geom(geom2)
            event = self._contact_event(body1_name, body2_name, geom1_name, geom2_name, contact)
            if event is None:
                continue
            key = (event["code"], event.get("human", ""), event.get("other_body", ""))
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
        return events

    def _contact_event(
        self,
        body1_name: str,
        body2_name: str,
        geom1_name: str,
        geom2_name: str,
        contact: Any,
    ) -> JsonDict | None:
        role1 = self._contact_role(body1_name)
        role2 = self._contact_role(body2_name)
        if role1 == "human" and role2 == "robot":
            return self._robot_human_contact_event(body1_name, body2_name, geom1_name, geom2_name, contact)
        if role1 == "robot" and role2 == "human":
            return self._robot_human_contact_event(body2_name, body1_name, geom2_name, geom1_name, contact)
        if role1 == "human" and role2 == "object":
            return self._object_human_contact_event(body1_name, body2_name, geom1_name, geom2_name, contact)
        if role1 == "object" and role2 == "human":
            return self._object_human_contact_event(body2_name, body1_name, geom2_name, geom1_name, contact)
        return None

    def _robot_human_contact_event(
        self,
        human_name: str,
        robot_body: str,
        human_geom: str,
        robot_geom: str,
        contact: Any,
    ) -> JsonDict:
        moving = self.world.robot.mode not in {"stopped", "refused", "waiting"}
        return {
            "type": "safety_violation" if moving else "safety_alert",
            "code": "robot_contact_human" if moving else "stationary_robot_contact_human",
            "message": "MuJoCo contact detected between the KUKA collision model and a human proxy."
            if moving
            else "MuJoCo contact detected between a stationary KUKA body and a human proxy.",
            "human": human_name,
            "other_body": robot_body,
            "human_geom": human_geom,
            "other_geom": robot_geom,
            "distance_m": round(float(contact.dist), 6),
            "position_m": self._contact_position(contact),
        }

    def _object_human_contact_event(
        self,
        human_name: str,
        object_name: str,
        human_geom: str,
        object_geom: str,
        contact: Any,
    ) -> JsonDict:
        obj = self.world.objects.get(object_name)
        hazardous = object_name == "knife" or bool(obj and obj.metadata.get("sharp"))
        return {
            "type": "safety_violation" if hazardous else "safety_alert",
            "code": "hazardous_object_contact_human" if hazardous else "object_contact_human",
            "message": "MuJoCo contact detected between a hazardous object and a human proxy."
            if hazardous
            else "MuJoCo contact detected between an object and a human proxy.",
            "human": human_name,
            "other_body": object_name,
            "human_geom": human_geom,
            "other_geom": object_geom,
            "distance_m": round(float(contact.dist), 6),
            "position_m": self._contact_position(contact),
        }

    def _body_name_for_geom(self, geom_id: int) -> str:
        import mujoco

        if self._mujoco_model is None:
            return ""
        body_id = int(self._mujoco_model.geom_bodyid[geom_id])
        return mujoco.mj_id2name(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"

    def _contact_role(self, body_name: str) -> str:
        if body_name in {human.id for human in self.world.humans}:
            return "human"
        if body_name in self.world.objects:
            return "object"
        if body_name in self._robot_body_names:
            return "robot"
        return "other"

    def _contact_position(self, contact: Any) -> list[float]:
        return [round(float(value), 6) for value in contact.pos]

    def _forward_mujoco(self) -> None:
        import mujoco

        if self._mujoco_model is not None and self._mujoco_data is not None:
            mujoco.mj_forward(self._mujoco_model, self._mujoco_data)

    def _refresh_robot_body_names(self) -> None:
        import mujoco

        if self._mujoco_model is None:
            return
        object_names = set(self.world.objects)
        human_names = set(self._physics_human_ids)
        names: set[str] = set()
        for body_id in range(1, self._mujoco_model.nbody):
            name = mujoco.mj_id2name(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name not in object_names and name not in human_names:
                names.add(name)
        self._robot_body_names = names

    def _all_scenario_human_ids(self) -> list[str]:
        ids: list[str] = []
        for human in self.scenario.humans:
            if human.id not in ids:
                ids.append(human.id)
        for entry in self.scenario.human_timeline:
            for human in entry.humans:
                if human.id not in ids:
                    ids.append(human.id)
        return ids

    def _initial_physics_humans(self) -> list:
        current = {human.id: human for human in self.scenario.humans_at(0)}
        all_humans: dict[str, Any] = {}
        for human in self.scenario.humans:
            all_humans.setdefault(human.id, human)
        for entry in self.scenario.human_timeline:
            for human in entry.humans:
                all_humans.setdefault(human.id, human)
        return [current.get(human_id, all_humans[human_id]) for human_id in self._physics_human_ids]

    def _kuka_joint_positions(self) -> dict[str, float]:
        if self._mujoco_model is None or self._mujoco_data is None:
            return {}
        return read_joint_positions(self._mujoco_model, self._mujoco_data)

    def _kuka_joint_targets(self) -> dict[str, float]:
        if self._mujoco_model is None or self._mujoco_data is None:
            return {}
        if self._last_joint_targets:
            return dict(self._last_joint_targets)
        return read_joint_controls(self._mujoco_model, self._mujoco_data)

    def _mujoco_time_s(self) -> float:
        if self._mujoco_data is None:
            return 0.0
        return round(float(self._mujoco_data.time), 6)

    def _mujoco_feedback(self) -> JsonDict:
        return {
            "physics": True,
            "time_s": self._mujoco_time_s(),
            "substeps": self._mujoco_substeps,
            "joint_positions": self._kuka_joint_positions(),
            "joint_targets": self._kuka_joint_targets(),
        }
