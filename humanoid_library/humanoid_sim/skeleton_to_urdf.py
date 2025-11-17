import numpy as np

class SkeletonToURDFMapper:
    def __init__(self):
        self.kp = {
            "nose": 0, "neck": 1,
            "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
            "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
            "mid_hip": 8,
            "r_hip": 9, "r_knee": 10, "r_ankle": 11,
            "l_hip": 12, "l_knee": 13, "l_ankle": 14
        }

        self.joint_names = [
            "spine",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
            "right_shoulder", "right_elbow",
            "left_shoulder", "left_elbow"
        ]

    # project a 3D vector into a 2D plane for angle calc
    def project_to_plane(self, v, plane):
        # v is (3,) in [x, y, z] (MediaPipe: x right, y down, z toward camera (neg = closer))
        if plane == "yz":    # sagittal: vertical (y) vs depth (z)
            return np.array([v[1], v[2]])
        if plane == "xy":    # image plane
            return np.array([v[0], v[1]])
        if plane == "xz":    # horizontal-depth
            return np.array([v[0], v[2]])
        raise ValueError("unknown plane")

    def angle_between_2d(self, a, b):
        # signed angle from a to b, both 2D vectors
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        cosang = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        ang = np.arccos(cosang)
        cross = a_norm[0] * b_norm[1] - a_norm[1] * b_norm[0]
        return ang if cross >= 0 else -ang

    def get_point3(self, skeleton, idx):
        # skeleton row: [x,y,z,vis]
        return skeleton[idx][:3]

    def skeleton_to_joint_angles(self, skeleton):
        kp = self.kp
        angles = {}

        # SPINE: use vector neck - mid_hip. compute rotation around camera axis (yaw-ish)
        neck = self.get_point3(skeleton, kp["neck"])
        mid_hip = self.get_point3(skeleton, kp["mid_hip"])
        spine_vec = neck - mid_hip
        # project into image plane (x horizontal, y vertical) to get tilt around z
        spine_2d = self.project_to_plane(spine_vec, "xy")
        # compute angle w.r.t. vertical (0, -1) because y is down in image coords
        vertical = np.array([0, -1.0])
        angles["spine"] = self.angle_between_2d(vertical, spine_2d)

        # RIGHT LEG: hip (thigh flexion), knee (bend), ankle (pitch)
        r_hip_p = self.get_point3(skeleton, kp["r_hip"])
        r_knee_p = self.get_point3(skeleton, kp["r_knee"])
        r_ankle_p = self.get_point3(skeleton, kp["r_ankle"])

        thigh = r_knee_p - r_hip_p           # thigh vector
        shin = r_ankle_p - r_knee_p          # shin vector

        # Hip flexion: project thigh on sagittal (y,z) and measure from vertical (y)
        hip_2d = self.project_to_plane(thigh, "yz")
        angles["right_hip"] = self.angle_between_2d(np.array([ -1.0, 0.0 ]), hip_2d)
        # note: I use [-1,0] meaning "up" in yz-plane if y is vertical-down and z depth

        # Knee: angle between thigh and shin in sagittal plane
        angles["right_knee"] = np.pi - self.angle_between_2d(
            self.project_to_plane(thigh, "yz"), self.project_to_plane(shin, "yz")
        )
        angles["right_knee"] = float(np.clip(angles["right_knee"], 0.0, np.pi))

        # Ankle: foot pitch relative to shin (project foot direction)
        # For simplicity, estimate foot vector as from ankle to toe (we may not have toe here; use small)
        # We'll compute ankle pitch as angle between shin and ankle->(ankle + small z offset)
        ankle_pitch = self.angle_between_2d(self.project_to_plane(shin, "yz"),
                                            self.project_to_plane(np.array([0.0,  -0.1, 0.0]) + shin, "yz"))
        angles["right_ankle"] = ankle_pitch

        # LEFT LEG (same)
        l_hip_p = self.get_point3(skeleton, kp["l_hip"])
        l_knee_p = self.get_point3(skeleton, kp["l_knee"])
        l_ankle_p = self.get_point3(skeleton, kp["l_ankle"])

        l_thigh = l_knee_p - l_hip_p
        l_shin = l_ankle_p - l_knee_p

        angles["left_hip"] = self.angle_between_2d(np.array([ -1.0, 0.0 ]), self.project_to_plane(l_thigh, "yz"))
        angles["left_knee"] = np.pi - self.angle_between_2d(
            self.project_to_plane(l_thigh, "yz"), self.project_to_plane(l_shin, "yz")
        )
        angles["left_knee"] = float(np.clip(angles["left_knee"], 0.0, np.pi))
        angles["left_ankle"] = self.angle_between_2d(self.project_to_plane(l_shin, "yz"),
                                                     self.project_to_plane(np.array([0.0, -0.1, 0.0]) + l_shin, "yz"))

        # RIGHT ARM: shoulder pitch (sagittal), elbow bend
        r_shoulder_p = self.get_point3(skeleton, kp["r_shoulder"])
        r_elbow_p = self.get_point3(skeleton, kp["r_elbow"])
        r_wrist_p = self.get_point3(skeleton, kp["r_wrist"])

        upper_arm = r_elbow_p - r_shoulder_p
        forearm = r_wrist_p - r_elbow_p

        # Shoulder pitch: project into sagittal
        angles["right_shoulder"] = self.angle_between_2d(np.array([ -1.0, 0.0 ]), self.project_to_plane(upper_arm, "yz"))
        # Elbow: bend between upper arm and forearm
        angles["right_elbow"] = np.pi - self.angle_between_2d(self.project_to_plane(upper_arm, "yz"),
                                                              self.project_to_plane(forearm, "yz"))
        angles["right_elbow"] = float(np.clip(angles["right_elbow"], 0.0, np.pi))

        # LEFT ARM
        l_shoulder_p = self.get_point3(skeleton, kp["l_shoulder"])
        l_elbow_p = self.get_point3(skeleton, kp["l_elbow"])
        l_wrist_p = self.get_point3(skeleton, kp["l_wrist"])

        l_upper_arm = l_elbow_p - l_shoulder_p
        l_forearm = l_wrist_p - l_elbow_p

        angles["left_shoulder"] = self.angle_between_2d(np.array([ -1.0, 0.0 ]), self.project_to_plane(l_upper_arm, "yz"))
        angles["left_elbow"] = np.pi - self.angle_between_2d(self.project_to_plane(l_upper_arm, "yz"),
                                                             self.project_to_plane(l_forearm, "yz"))
        angles["left_elbow"] = float(np.clip(angles["left_elbow"], 0.0, np.pi))

        # Return angles (radians) dictionary
        return angles

    def angles_to_action_vector(self, angles):
        return np.array([angles.get(name, 0.0) for name in self.joint_names], dtype=np.float32)
