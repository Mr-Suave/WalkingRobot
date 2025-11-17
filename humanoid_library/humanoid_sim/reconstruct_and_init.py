# reconstruct_and_init.py
import numpy as np
import cv2
import pybullet as p
from math import radians

# canonical 3D template for the 25 keypoints (meters).
# Coordinates are in a simple model frame:
# x: right, y: up, z: forward (towards camera) - we will adapt to camera later.
# These are rough anthropometric proportions and match your kp mapping.
# Index mapping: 0 nose, 1 neck, 2 r_sh, 3 r_el, 4 r_wr, 5 l_sh, 6 l_el, 7 l_wr,
# 8 mid_hip, 9 r_hip, 10 r_knee, 11 r_ankle, 12 l_hip, 13 l_knee, 14 l_ankle, ...
CANONICAL_3D = {
    0: np.array([0.0, 1.55, 0.0]),     # nose (top-ish)
    1: np.array([0.0, 1.45, 0.0]),     # neck
    2: np.array([0.18, 1.4, 0.0]),     # r_shoulder
    3: np.array([0.42, 1.05, 0.0]),    # r_elbow
    4: np.array([0.62, 0.85, 0.0]),    # r_wrist
    5: np.array([-0.18, 1.4, 0.0]),    # l_shoulder
    6: np.array([-0.42, 1.05, 0.0]),   # l_elbow
    7: np.array([-0.62, 0.85, 0.0]),   # l_wrist
    8: np.array([0.0, 1.0, 0.0]),      # mid_hip
    9: np.array([0.12, 0.98, 0.0]),    # r_hip
    10: np.array([0.12, 0.6, 0.0]),    # r_knee
    11: np.array([0.12, 0.1, 0.05]),   # r_ankle (slightly forward)
    12: np.array([-0.12, 0.98, 0.0]),  # l_hip
    13: np.array([-0.12, 0.6, 0.0]),   # l_knee
    14: np.array([-0.12, 0.1, 0.05]),  # l_ankle
    15: np.array([0.05, 1.6, 0.02]),   # r_eye
    16: np.array([-0.05, 1.6, 0.02]),  # l_eye
    17: np.array([0.12, 1.55, -0.05]), # r_ear
    18: np.array([-0.12, 1.55, -0.05]),# l_ear
    19: np.array([-0.05, 0.0, 0.12]),  # l_big_toe (not used heavily)
    20: np.array([-0.02, 0.0, 0.08]),  # l_small_toe
    21: np.array([-0.10, 0.0, -0.02]), # l_heel
    22: np.array([0.05, 0.0, 0.12]),   # r_big_toe
    23: np.array([0.02, 0.0, 0.08]),   # r_small_toe
    24: np.array([0.10, 0.0, -0.02]),  # r_heel
}

# default camera intrinsics fallback if user does not provide
def default_camera_matrix(image_width, image_height):
    # focal length ~ 0.9 * width (reasonable for many consumer cameras)
    fx = fy = 0.9 * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def build_canonical_points(scale=1.0):
    """Return an Nx3 array of canonical joints scaled by scale (meters) in the same 0..24 indices."""
    pts = []
    for i in range(25):
        v = CANONICAL_3D.get(i, np.array([0.0, 1.0, 0.0]))
        pts.append(v * scale)
    return np.array(pts, dtype=np.float64)  # shape (25,3)

def normalized_to_pixel(skeleton25, image_width, image_height):
    """Convert MediaPipe normalized [x,y] to pixel coordinates."""
    pts2d = []
    for i in range(25):
        x, y = skeleton25[i][0], skeleton25[i][1]  # normalized
        px = float(x) * image_width
        py = float(y) * image_height
        pts2d.append([px, py])
    return np.array(pts2d, dtype=np.float64)

def solve_pnp_and_reconstruct(skeleton25, image_wh,
                              camera_matrix=None, dist_coeffs=None,
                              person_height_m=1.75):
    """
    Reconstruct consistent 3D joint positions (camera frame) from one image.
    Args:
        skeleton25: (25,4) array [x_norm, y_norm, z?, vis]
        image_wh: (width, height)
        camera_matrix: 3x3 (optional). If None, a default heuristic matrix is used.
        dist_coeffs: distortion coefficients (optional).
        person_height_m: expected real height (meters) to scale canonical model.
    Returns:
        joint_positions_cam: (25,3) np.array in camera coords (meters)
        rvec, tvec: pose of canonical model in camera frame
    """
    w, h = image_wh
    if camera_matrix is None:
        camera_matrix = default_camera_matrix(w, h)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))

    # 2D image points (use only the reliable subset for PnP)
    image_points = normalized_to_pixel(skeleton25, w, h)

    # Build canonical 3D and scale it so canonical height matches person_height_m
    canonical_points = build_canonical_points(scale=1.0)
    # canonical height estimated from model: nose (y) - ankle (y)
    # compute model height and compute scale to match desired person_height_m
    ys = canonical_points[:,1]
    model_height = np.max(ys) - np.min(ys)
    if model_height <= 0:
        scale = 1.0
    else:
        scale = person_height_m / model_height
    canonical_points *= scale

    # Now select corresponding indices used in PnP. Use many stable correspondences:
    idxs_for_pnp = [1, 2, 5, 8, 9, 12, 3, 6, 10, 13, 11, 14, 4, 7]  # neck, shoulders, hips, elbows, knees, ankles, wrists
    obj_pts = np.array([canonical_points[i] for i in idxs_for_pnp], dtype=np.float64)
    img_pts = np.array([image_points[i] for i in idxs_for_pnp], dtype=np.float64)

    # Solve PnP
    # initial guess via SOLVEPNP_EPNP then refine
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        # fallback try iterative
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("solvePnP failed")

    # Transform canonical_points (model) into camera frame: X_cam = R * X_model + t
    Rm, _ = cv2.Rodrigues(rvec)
    model_in_cam = (Rm @ canonical_points.T).T + tvec.ravel()

    return model_in_cam, (rvec, tvec), camera_matrix, canonical_points

def enforce_bone_lengths(target_positions, canonical_positions, kp_parent_child_pairs=None):
    """
    Enforce bone lengths from canonical skeleton onto target positions.
    target_positions: (25,3) positions (camera frame)
    canonical_positions: (25,3) canonical positions (already scaled to person)
    kp_parent_child_pairs: list of (parent_idx, child_idx) bones
    Returns:
        corrected_positions (25,3)
    """
    corrected = target_positions.copy()
    if kp_parent_child_pairs is None:
        kp_parent_child_pairs = [
            (1,2),(2,3),(3,4),      # right arm: neck->r_sh->r_el->r_wr
            (1,5),(5,6),(6,7),      # left arm
            (8,9),(9,10),(10,11),   # right leg: mid_hip->r_hip->r_knee->r_ankle
            (8,12),(12,13),(13,14), # left leg
            (8,1)                   # mid_hip->neck (spine)
        ]
    # compute canonical bone lengths
    canonical_lengths = {}
    for p,c in kp_parent_child_pairs:
        canonical_lengths[(p,c)] = np.linalg.norm(canonical_positions[c] - canonical_positions[p]) + 1e-8

    # enforce length by walking bones: parent -> child
    # we'll iterate a few times to converge
    for _ in range(3):
        for (p,c), L in canonical_lengths.items():
            vec = corrected[c] - corrected[p]
            n = np.linalg.norm(vec) + 1e-8
            corrected[c] = corrected[p] + (vec / n) * L

    return corrected

def apply_initial_pose_to_pybullet(env, joint_positions_cam, use_ik=True):
    """
    env: instance of your HumanoidWalkingEnv (already created)
    joint_positions_cam: (25,3) in camera frame (meters)
    This function computes IK for feet and hands and sets initial joint states.
    """
    # Map the body part target positions to world positions for pybullet.
    # We assume camera frame maps to world frame with a translation: place pelvis at x=0,y=0,z=target_height
    # We'll choose a placement that puts mid_hip at target_z above ground.
    mid_hip_idx = 8
    mid_hip_cam = joint_positions_cam[mid_hip_idx]

    # Determine placement: we want pelvis (mid_hip) to be at world z = target_height (~0.9)
    target_pelvis_height = 0.9
    dz = target_pelvis_height - mid_hip_cam[1]  # because canonical y=up while camera may be different; we will treat cam y as up
    # Simple mapping: camera X->world X, camera Z->world Y (forward), camera Y->world Z (up)
    def cam_to_world(pt):
        # convert: cam [x, y, z] -> world [x, z, y]
        return [pt[0], pt[2], pt[1] + dz]

    # compute world targets for feet and wrists
    r_ank = cam_to_world(joint_positions_cam[11])
    l_ank = cam_to_world(joint_positions_cam[14])
    r_wrist = cam_to_world(joint_positions_cam[4])
    l_wrist = cam_to_world(joint_positions_cam[7])
    pelvis_world = cam_to_world(joint_positions_cam[mid_hip_idx])

    # Position the whole body such that pelvis is at pelvis_world
    # Reset base pose:
    p.resetBasePositionAndOrientation(env.humanoid_id, pelvis_world, p.getQuaternionFromEuler([0,0,0]))

    # IK target link names in your URDF: use right_foot (link name "right_foot"), left_foot, right_forearm? We will query link indices.
    # Find link indices
    name_to_idx = {}
    for i in range(p.getNumJoints(env.humanoid_id)):
        ji = p.getJointInfo(env.humanoid_id, i)
        name = ji[12].decode('utf-8')  # link name
        name_to_idx[name] = i

    # Common link names in your URDF: right_foot, left_foot, right_forearm, left_forearm
    ik_targets = []
    if "right_foot" in name_to_idx:
        ik_targets.append(("right_foot", r_ank))
    if "left_foot" in name_to_idx:
        ik_targets.append(("left_foot", l_ank))
    # For hands, pin to wrists by using the end link of forearm->foot? If not present, skip
    if "right_forearm" in name_to_idx:
        ik_targets.append(("right_forearm", r_wrist))
    if "left_forearm" in name_to_idx:
        ik_targets.append(("left_forearm", l_wrist))

    # Solve IK per target. We will iterate and set joint states (this is a heuristic but works well for initial pose)
    solved_joint_angles = {}
    for link_name, target in ik_targets:
        link_index = name_to_idx[link_name]
        # PyBullet expects targetPos in world coordinates
        ik_joints = p.calculateInverseKinematics(env.humanoid_id, link_index, target)
        # ik_joints is length = num_dofs of robot; set relevant joints
        # We'll set every movable joint to the IK solution
        for j_idx, q in enumerate(ik_joints):
            # map j_idx -> joint index in pybullet (same ordering)
            # For safety, clamp to joint limits
            try:
                info = p.getJointInfo(env.humanoid_id, j_idx)
            except Exception:
                continue
            lower = info[8]; upper = info[9]
            if lower < upper:
                q_clamped = float(np.clip(q, lower, upper))
            else:
                q_clamped = float(q)
            p.resetJointState(env.humanoid_id, j_idx, q_clamped)

    # Finally, run a few simulation steps to settle
    for _ in range(50):
        p.stepSimulation()

    # If you want, return current joint positions
    joint_positions = []
    for name in env.joint_names:
        if name in env.joint_indices:
            s = p.getJointState(env.humanoid_id, env.joint_indices[name])
            joint_positions.append(s[0])
        else:
            joint_positions.append(0.0)
    return np.array(joint_positions, dtype=np.float32)

class Reconstructor:
    def __init__(self, person_height_m=1.75, camera_matrix=None, dist_coeffs=None):
        self.person_height_m = person_height_m
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def process(self, body25, image_wh):
        """
        body25: (25,4) mediapipe or openpose body25
        image_wh: (width, height)
        Returns:
            corrected_3d   → final 3D skeleton in camera coords
            canonical_3d   → canonical model after scaling
            pnp_meta       → (rvec, tvec, K)
        """

        # Step 1: Solve PnP → get raw 3D
        joint3d, (rvec, tvec), K, canonical_points = solve_pnp_and_reconstruct(
            body25,
            image_wh,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            person_height_m=self.person_height_m
        )

        # Step 2: Clean bone lengths
        corrected = enforce_bone_lengths(joint3d, canonical_points)

        meta = {
            "rvec": rvec,
            "tvec": tvec,
            "K": K
        }

        return corrected, canonical_points, meta

    def apply_to_env(self, env, joint3d):
        """Calls the provided IK + base pose setter."""
        return apply_initial_pose_to_pybullet(env, joint3d)
