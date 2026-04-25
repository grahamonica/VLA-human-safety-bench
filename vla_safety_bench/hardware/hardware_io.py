from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from vla_safety_bench.types import JsonDict, Vector3


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fovy_deg: float


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in world frame.

    rotation_matrix is a 3x3 row-major matrix whose columns are the camera-frame
    basis vectors expressed in world coordinates: column 0 = camera +x (image right),
    column 1 = camera +y (image up), column 2 = camera +z (toward viewer; the optical
    axis points along -z).
    """

    position_m: Vector3
    rotation_matrix: tuple[Vector3, Vector3, Vector3]

    def x_axis(self) -> Vector3:
        return (self.rotation_matrix[0][0], self.rotation_matrix[1][0], self.rotation_matrix[2][0])

    def y_axis(self) -> Vector3:
        return (self.rotation_matrix[0][1], self.rotation_matrix[1][1], self.rotation_matrix[2][1])


class HardwareIO(Protocol):
    """Adapter between the harness and a physical robot.

    Implementations must return RGB frames as uint8 HxWx3 numpy arrays. Camera pose
    is in world (robot-base) frame; rotation_matrix follows the convention in
    `CameraPose`. `submit_action` forwards the VLA's action to the controller and
    returns whatever feedback the controller produced (joint state, fault codes,
    etc.) — this is recorded in the trace but not used for scoring.
    """

    def read_camera_frame(self, camera_name: str) -> Any:
        ...

    def read_camera_intrinsics(self, camera_name: str) -> CameraIntrinsics:
        ...

    def read_camera_pose(self, camera_name: str) -> CameraPose:
        ...

    def read_robot_state(self) -> JsonDict:
        ...

    def submit_action(self, action: JsonDict) -> JsonDict:
        ...


@dataclass
class MockHardwareIO:
    """In-process HardwareIO for unit tests and dry runs without a robot.

    Returns a fixed solid-colour frame and a fixed wrist-cam pose. Records every
    submitted action so tests can assert on the sequence sent to the "hardware".
    """

    width: int = 640
    height: int = 480
    fovy_deg: float = 92.0
    background_rgb: tuple[int, int, int] = (200, 205, 210)
    camera_position_m: Vector3 = (0.30, 0.0, 0.55)
    camera_rotation_matrix: tuple[Vector3, Vector3, Vector3] = (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    submitted_actions: list[JsonDict] = field(default_factory=list)
    robot_state: JsonDict = field(default_factory=dict)

    def read_camera_frame(self, camera_name: str) -> Any:
        import numpy as np

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = self.background_rgb[0]
        frame[:, :, 1] = self.background_rgb[1]
        frame[:, :, 2] = self.background_rgb[2]
        return frame

    def read_camera_intrinsics(self, camera_name: str) -> CameraIntrinsics:
        return CameraIntrinsics(width=self.width, height=self.height, fovy_deg=self.fovy_deg)

    def read_camera_pose(self, camera_name: str) -> CameraPose:
        return CameraPose(position_m=self.camera_position_m, rotation_matrix=self.camera_rotation_matrix)

    def read_robot_state(self) -> JsonDict:
        return dict(self.robot_state)

    def submit_action(self, action: JsonDict) -> JsonDict:
        self.submitted_actions.append(dict(action))
        return {"accepted": True, "action_index": len(self.submitted_actions) - 1}
