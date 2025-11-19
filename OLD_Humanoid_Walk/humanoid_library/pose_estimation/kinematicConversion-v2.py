import numpy as np

# BODY25 index map using your custom indices
B = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,
    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
    "mid_hip": 8,
    "r_hip": 9,
    "r_knee": 10,
    "r_ankle": 11,
    "l_hip": 12,
    "l_knee": 13,
    "l_ankle": 14
}

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-6)

def angle_between(v1, v2):
    v1 = unit(v1); v2 = unit(v2)
    dot = np.clip(np.dot(v1, v2), -1, 1)
    return np.arccos(dot)

# ----------------- torso angles -----------------
def compute_spine_angles(neck, hip):
    spine = unit(neck - hip)
    yaw   = np.arctan2(spine[0], spine[2])
    pitch = -np.arctan2(spine[1], spine[2])
    roll  = np.arctan2(spine[1], spine[0])
    return yaw, pitch, roll

# ----------------- leg IK -----------------
def leg_angles(hip, knee, ankle):
    thigh = knee - hip
    shin  = ankle - knee
    t = unit(thigh)

    hip_roll  = np.arctan2(t[1], t[0])
    hip_yaw   = np.arctan2(t[0], t[2])
    hip_pitch = np.arctan2(-t[1], t[2])

    knee_angle = angle_between(thigh, shin)
    return hip_roll, hip_yaw, hip_pitch, knee_angle

# ----------------- arm IK -----------------
def arm_angles(shoulder, elbow, wrist):
    upper = elbow - shoulder
    lower = wrist - elbow
    u = unit(upper)

    shoulder_pitch = np.arctan2(-u[1], u[2])
    shoulder_roll  = np.arctan2(u[0], u[2])
    elbow_angle = angle_between(upper, lower)
    return shoulder_pitch, shoulder_roll, elbow_angle

# --------------- MAIN CONVERSION -----------------
def body25_to_humanoid_pose(body25):
    # torso
    yaw, pitch, roll = compute_spine_angles(
        body25[B["neck"]], 
        body25[B["mid_hip"]]
    )

    # right leg
    r_hr, r_hz, r_hy, r_k = leg_angles(
        body25[B["r_hip"]],
        body25[B["r_knee"]],
        body25[B["r_ankle"]]
    )

    # left leg
    l_hr, l_hz, l_hy, l_k = leg_angles(
        body25[B["l_hip"]],
        body25[B["l_knee"]],
        body25[B["l_ankle"]]
    )

    # right arm
    r_s1, r_s2, r_e = arm_angles(
        body25[B["r_shoulder"]],
        body25[B["r_elbow"]],
        body25[B["r_wrist"]]
    )

    # left arm
    l_s1, l_s2, l_e = arm_angles(
        body25[B["l_shoulder"]],
        body25[B["l_elbow"]],
        body25[B["l_wrist"]]
    )

    # MUJOCO HUMANOID JOINT ORDER
    return np.array([
        yaw, pitch, roll,      # abdomen z,y,x
        r_hr, r_hz, r_hy, r_k, # right hip x,z,y,knee
        l_hr, l_hz, l_hy, l_k, # left hip x,z,y,knee
        r_s1, r_s2, r_e,       # right shoulder1,2, elbow
        l_s1, l_s2, l_e
    ], dtype=np.float32)
