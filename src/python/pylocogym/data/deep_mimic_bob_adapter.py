from dataclasses import dataclass
from enum import auto
from typing import ClassVar, Dict, Tuple

import numpy as np
from pylocogym.data.dataset import (
    Fields,
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
    StrEnum,
)
from pylocogym.data.deep_mimic_motion import DeepMimicMotion
from scipy.spatial.transform import Rotation as R


def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array(
        [
            q1[0] * q2[1] - q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2],
            q1[0] * q2[2] + q1[1] * q2[3] - q1[2] * q2[0] - q1[3] * q2[1],
            q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1] - q1[3] * q2[0],
        ]
    )


class BobMotionDataFieldNames(StrEnum):
    """
    Enum class for Bob motion data field names.
    """

    ROOT_POS = auto()
    ROOT_ROT = auto()
    JOINTS = auto()


class BobMotionDataField(Fields):
    FieldNames = BobMotionDataFieldNames
    fields: Dict[BobMotionDataFieldNames, Tuple[int, int]] = {
        BobMotionDataFieldNames.ROOT_POS: (0, 3),
        BobMotionDataFieldNames.ROOT_ROT: (3, 6),
        BobMotionDataFieldNames.JOINTS: (6, 50),
    }


@dataclass
class BobMotionDataSample(MotionDataSample):
    FieldsType: ClassVar = BobMotionDataField


@dataclass
class BobKeyframeMotionDataSample(KeyframeMotionDataSample):
    FieldsType: ClassVar = BobMotionDataField
    BaseSampleType: ClassVar = BobMotionDataSample


class BobMotionBobAdapter(MapKeyframeMotionDataset):
    SampleType = BobKeyframeMotionDataSample
    mimic_joints_index = {  0,3,6,18,23,26,27,30,33,28,
                            31,34,13,16,14,17,1,4,7,2,
                            5,8,36,37,10,11}

    def __init__(
        self,
        in_dataset: DeepMimicMotion,
        num_joints,
        joint_lower_limit,
        joint_upper_limit,
        joint_default_angle=None,
        rescale=False,
    ):
        self.in_dataset = in_dataset
        self.joint_angle_limit_low = joint_lower_limit
        self.joint_angle_limit_high = joint_upper_limit
        self.joint_angle_default = joint_default_angle
        if self.joint_angle_default is None:
            self.joint_angle_default = np.zeros(num_joints)
        self.joint_scale_factors = np.maximum(
            abs(self.joint_angle_default - self.joint_angle_limit_low),
            abs(self.joint_angle_default - self.joint_angle_limit_high),
        )
        self.num_joints = num_joints
        self.is_rescale_action = rescale

    def quart_to_rpy(self, q, mode):
        # q is in (w,x,y,z) format
        q_xyzw = list(q[1:])
        q_xyzw.append(q[0])
        r = R.from_quat(q_xyzw)
        euler = r.as_euler(mode)
        return euler[0], euler[1], euler[2]

    def rescale_action(self, action):
        bound_action = np.minimum(np.maximum(action, self.joint_angle_limit_low), self.joint_angle_limit_high)
        scaled_action = (bound_action - self.joint_angle_default) / self.joint_scale_factors
        return scaled_action

    def retarget_base_orientation(self, motion_clips_q):
        (yaw, pitch, roll)  = self.quart_to_rpy(motion_clips_q[3:7], 'yzx')
        return yaw, -pitch, roll

    def retarget_joint_angle(self, motion_clips_q):
        """Given a motion_clips orientation data, return a retarget action"""
        joints = np.zeros(self.num_joints)

        (chest_z, chest_y, chest_x) = self.quart_to_rpy(motion_clips_q[7:11], "zyx")
        (neck_z, neck_y, neck_x) = self.quart_to_rpy(motion_clips_q[11:15], "zyx")
        (r_hip_z, r_hip_x, r_hip_y) = self.quart_to_rpy(motion_clips_q[15:19], "zxy")
        (r_ankle_z, r_ankle_x, r_ankle_y) = self.quart_to_rpy(motion_clips_q[20:24], "zxy")
        (r_shoulder_z, r_shoulder_x, r_shoulder_y) = self.quart_to_rpy(motion_clips_q[24:28], "zxy")
        (l_hip_z, l_hip_x, l_hip_y) = self.quart_to_rpy(motion_clips_q[29:33], "zxy")
        (l_ankle_z, l_ankle_x, l_ankle_y) = self.quart_to_rpy(motion_clips_q[34:38], "zxy")
        (l_shoulder_z, l_shoulder_x, l_shoulder_y) = self.quart_to_rpy(motion_clips_q[38:42], "zxy")

        # chest - xyz euler angle
        joints[0] = -chest_z
        joints[3] = chest_y
        joints[6] = chest_x

        # neck - xyz euler angle
        joints[18] = -neck_z
        joints[23] = neck_y
        joints[26] = neck_x

        # shoulder - xzy euler angle
        joints[27] = -l_shoulder_z
        joints[30] = l_shoulder_x
        joints[33] = l_shoulder_y

        joints[28] = -r_shoulder_z
        joints[31] = r_shoulder_x
        joints[34] = r_shoulder_y

        # ankle - xzy euler angle
        joints[13] = -l_ankle_z
        joints[16] = l_ankle_x

        joints[14] = -r_ankle_z
        joints[17] = r_ankle_x

        # hip - xzy euler angle
        joints[1] = -l_hip_z
        joints[4] = l_hip_x
        joints[7] = l_hip_y

        joints[2] = -r_hip_z
        joints[5] = r_hip_x
        joints[8] = r_hip_y

        r_knee = motion_clips_q[19:20]
        r_elbow = motion_clips_q[28:29]
        l_knee = motion_clips_q[33:34]
        l_elbow = motion_clips_q[42:43]

        # elbow - revolute joint
        joints[36] = l_elbow
        joints[37] = r_elbow

        # knee - revolute joint
        joints[10] = l_knee
        joints[11] = r_knee

        if self.is_rescale_action is True:
            joints = self.rescale_action(joints)
        else:
            joints = np.minimum(np.maximum(joints, self.joint_angle_limit_low), self.joint_angle_limit_high)

        return joints

    def __len__(self) -> int:
        return len(self.in_dataset)

    def __getitem__(self, idx):
        sample = self.in_dataset[idx]
        j0 = self.retarget_joint_angle(sample.q0)
        j1 = self.retarget_joint_angle(sample.q1)
        root_rot_0 = self.retarget_base_orientation(sample.q0)
        root_rot_1 = self.retarget_base_orientation(sample.q1)
        root_pos_0 = sample.q0_fields.root_pos
        root_pos_1 = sample.q1_fields.root_pos
        root_ang_vel = angular_velocities(sample.q0_fields.root_rot, sample.q1_fields.root_rot, sample.dt)
        q0 = np.concatenate([root_pos_0, root_rot_0, j0])
        q1 = np.concatenate([root_pos_1, root_rot_1, j1])
        qdot = (q1 - q0) / sample.dt
        qdot[3:6] = root_ang_vel

        return BobKeyframeMotionDataSample(
            t0=sample.t0,
            q0=q0,
            q1=q1,
            qdot=qdot,
            dt=sample.dt,
            phase0=sample.phase0,
            phase1=sample.phase1,
        )