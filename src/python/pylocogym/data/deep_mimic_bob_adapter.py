from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, Tuple, Union

import numpy as np
from pylocogym.data.dataset import (
    Fields,
    KeyframeMotionDataSample,
    MotionDataSample,
    StrEnum,
)
from pylocogym.data.deep_mimic_motion import DeepMimicKeyframeMotionDataSample, DeepMimicMotion, DeepMimicMotionDataSample
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


class DeepMimicMotionBobAdapter(DeepMimicMotion):
    SampleType = BobKeyframeMotionDataSample
    mimic_joints_index = {0, 3, 6, 18, 23, 26, 27, 30, 33, 28, 31, 34, 13, 16, 14, 17, 1, 4, 7, 2, 5, 8, 36, 37, 10, 11}

    def __init__(
        self,
        path: Union[str, Path],
        num_joints,
        joint_lower_limit,
        joint_upper_limit,
        joint_default_angle=None,
        rescale=False,
    ):
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
        #? Konstantinos: Why not use the command: `bound_action = np.clip(action, self.joint_angle_limit_low, self.[...]_high)`?
        scaled_action = (bound_action - self.joint_angle_default) / self.joint_scale_factors
        return scaled_action

    def retarget_base_orientation(self, motion_clips_q):
        (yaw, pitch, roll) = self.quart_to_rpy(motion_clips_q[3:7], "YZX")
        return yaw, -pitch, roll

    def retarget_joint_angle(self, motion_clips_q):
        """Given a motion_clips orientation data, return a retarget action"""
        joints = np.zeros(self.num_joints)

        (chest_z, chest_y, chest_x) = self.quart_to_rpy(motion_clips_q[7:11], "ZYX")
        (neck_z, neck_y, neck_x) = self.quart_to_rpy(motion_clips_q[11:15], "ZYX")
        (r_hip_z, r_hip_x, r_hip_y) = self.quart_to_rpy(motion_clips_q[15:19], "ZXY")
        (r_ankle_z, r_ankle_x, r_ankle_y) = self.quart_to_rpy(motion_clips_q[20:24], "ZXY")
        (r_shoulder_z, r_shoulder_x, r_shoulder_y) = self.quart_to_rpy(motion_clips_q[24:28], "ZXY")
        (l_hip_z, l_hip_x, l_hip_y) = self.quart_to_rpy(motion_clips_q[29:33], "ZXY")
        (l_ankle_z, l_ankle_x, l_ankle_y) = self.quart_to_rpy(motion_clips_q[34:38], "ZXY")
        (l_shoulder_z, l_shoulder_x, l_shoulder_y) = self.quart_to_rpy(motion_clips_q[38:42], "ZXY")

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
        joints[36] = -l_elbow
        joints[37] = -r_elbow

        # knee - revolute joint
        joints[10] = -l_knee
        joints[11] = -r_knee

        if self.is_rescale_action:
            joints = self.rescale_action(joints)
        else:
            joints = np.minimum(np.maximum(joints, self.joint_angle_limit_low), self.joint_angle_limit_high)
            #? Again, why not use the `np.clip` function, instead of min(max())?

        return joints

    def retarget_base_pos(self, motion_clips_q):
        # our's x axis == -1 * motion's z axis
        # our's z axis == motion's x axis
        return np.array([-motion_clips_q[2], motion_clips_q[1], motion_clips_q[0]])


    def adapt(self, sample: DeepMimicMotionDataSample, kf: DeepMimicKeyframeMotionDataSample) -> BobMotionDataSample:
        j = self.retarget_joint_angle(sample.q)
        root_rot = self.retarget_base_orientation(sample.q)
        root_pos = self.retarget_base_pos(sample.q)
        q = np.concatenate([root_pos, root_rot, j])

        j0 = self.retarget_joint_angle(kf.q0)
        j1 = self.retarget_joint_angle(kf.q1)
        root_rot_0 = self.retarget_base_orientation(kf.q0)
        root_rot_1 = self.retarget_base_orientation(kf.q1)
        root_pos_0 = self.retarget_base_pos(kf.q0)
        root_pos_1 = self.retarget_base_pos(kf.q1)
        q0 = np.concatenate([root_pos_0, root_rot_0, j0])
        q1 = np.concatenate([root_pos_1, root_rot_1, j1])
        qdot = (q1 - q0) / kf.dt
        root_ang_vel = angular_velocities(kf.q0_fields.root_rot, kf.q1_fields.root_rot, kf.dt)
        # our's y axis == motion's y axis
        # our's x axis == -1 * motion's z axis
        # our's z axis == motion's x axis
        qdot[3:6] = np.array([root_ang_vel[1], -root_ang_vel[2], root_ang_vel[0]])

        return BobMotionDataSample(
            t=sample.t,
            q=q,
            qdot=qdot,
            phase=sample.phase,
        )
