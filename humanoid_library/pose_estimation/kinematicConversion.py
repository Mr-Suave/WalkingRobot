import numpy as np
import cv2
import mediapipe as mp
from .keyPointExtraction import PoseExtractor
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

def select_main_skeleton_multiple(extractor, image, save_path):
    """
    Detect multiple people, extract skeletons, and choose the first non-empty one
    (prefers largest bounding boxes first).
    """

    from ultralytics import YOLO

    yolo_model = YOLO("yolov8m.pt")

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    results = yolo_model.predict(source=image, verbose=False)
    boxes = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
            else:
                print("class: ", cls)

    if not boxes:
        print("No person detected")
        return None

    # sort by area (largest first)
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    print(f"Detected {len(boxes)} person(s). Trying each until pose found...")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        person_crop = image[y1:y2, x1:x2]
        skeleton = extractor.extract_keypoints(person_crop)

        if skeleton is None or len(skeleton) == 0:
            print(f"Person {i+1}: No pose detected, Trying next...")
            continue

        print(f"Pose found for person {i+1}")
        # map coords back to original image
        skeleton[:, 0] = (x1 + skeleton[:, 0] * (x2 - x1)) / image.shape[1]
        skeleton[:, 1] = (y1 + skeleton[:, 1] * (y2 - y1)) / image.shape[0]

        extractor.draw_skeleton(image, skeleton, save_path)
        return skeleton

    print("No valid skeleton found in any person box")
    return None

def vector_to_rotation(parent_vec, child_vec, joint_axis=None):
    """
    Compute rotation to align parent_vec with child_vec.
    - parent_vec: np.array([x, y, z])
    - child_vec: np.array([x, y, z])
    - joint_axis: optional np.array([x, y, z]) for hinge joints
    Returns quaternion [x, y, z, w]
    """
    parent_vec = parent_vec / np.linalg.norm(parent_vec)
    child_vec = child_vec / np.linalg.norm(child_vec)

    if joint_axis is not None:
        # Hinge rotation: project child_vec onto plane orthogonal to axis
        axis = joint_axis / np.linalg.norm(joint_axis)
        proj = child_vec - np.dot(child_vec, axis) * axis
        angle = np.arctan2(np.linalg.norm(np.cross(parent_vec, proj)), np.dot(parent_vec, proj))
        return R.from_rotvec(angle * axis).as_quat()
    else:
        # Spherical joint: rotation from parent_vec to child_vec
        cross = np.cross(parent_vec, child_vec)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-6:  # vectors aligned
            return np.array([0, 0, 0, 1])
        dot = np.dot(parent_vec, child_vec)
        angle = np.arctan2(norm_cross, dot)
        axis = cross / norm_cross
        return R.from_rotvec(angle * axis).as_quat()



# def compute_joint_angles(skeleton):
#     """
#     Computes the 9 relative joint angles (in radians) needed by the simulation.
#     """
    
#     # Helper function to get a vector (x, y) from two keypoint indices
#     def get_vec(p1_idx, p2_idx):
#         p1 = skeleton[p1_idx][:2]
#         p2 = skeleton[p2_idx][:2]
#         return p2 - p1

#     # Helper function to get the signed angle between two vectors
#     def angle_between(v1, v2):
#         # Calculates the angle from v1 to v2
#         angle1 = np.arctan2(v1[1], v1[0])
#         angle2 = np.arctan2(v2[1], v2[0])
#         return angle2 - angle1

#     # Limb vectors
#     v_spine_up = get_vec(8, 1)     # MidHip -> Neck (our "up" reference)
#     v_spine_down = -v_spine_up     # A vector pointing "down" the spine
    
#     v_r_thigh = get_vec(9, 10)     # R Hip -> R Knee
#     v_r_calf = get_vec(10, 11)     # R Knee -> R Ankle
    
#     v_l_thigh = get_vec(12, 13)    # L Hip -> L Knee
#     v_l_calf = get_vec(13, 14)     # L Knee -> L Ankle
    
#     v_r_upper_arm = get_vec(2, 3)  # R Shoulder -> R Elbow
#     v_r_forearm = get_vec(3, 4)    # R Elbow -> R Wrist
    
#     v_l_upper_arm = get_vec(5, 6)  # L Shoulder -> L Elbow
#     v_l_forearm = get_vec(6, 7)    # L Elbow -> L Wrist
    
#     v_vertical_up = np.array([0, -1]) # Y=0 is top, so [0, -1] is UP

#     # --- Calculate Relative Joint Angles ---
    
#     # 1. R Hip
#     r_hip_angle = -angle_between(v_spine_down, v_r_thigh)
#     # 2. R Knee
#     r_knee_angle = angle_between(v_r_thigh, v_r_calf)
#     # 3. L Hip
#     l_hip_angle = angle_between(v_spine_down, v_l_thigh)
#     # 4. L Knee
#     l_knee_angle = angle_between(v_l_thigh, v_l_calf)
    
#     # 5. R Shoulder
#     r_shoulder_angle = angle_between(v_spine_down, v_r_upper_arm)
#     # 6. R Elbow (No negative sign)
#     r_elbow_angle = angle_between(v_r_upper_arm, v_r_forearm)
    
#     # 7. L Shoulder (Mirrored axis)
#     l_shoulder_angle = -angle_between(v_spine_down, v_l_upper_arm)
#     # 8. L Elbow (Mirrored axis)
#     l_elbow_angle = -angle_between(v_l_upper_arm, v_l_forearm)
    
#     # 9. Spine
#     spine_angle = angle_between(v_vertical_up, v_spine_up)

#     angles = np.array([
#         r_hip_angle, r_knee_angle,
#         l_hip_angle, l_knee_angle,
#         r_shoulder_angle, r_elbow_angle-90*np.pi/180,
#         l_shoulder_angle, l_elbow_angle-90*np.pi/180,
#         spine_angle
#     ], dtype=np.float32)
#     print(angles*180/np.pi)

#     return angles

def compute_joint_angles(skeleton):
    """
    Compute 3D rotations (quaternions) for a humanoid based on skeleton keypoints.
    Assumes skeleton is a Nx3 array: [x, y, z] for each keypoint.
    Returns a dict of quaternions per joint.
    """

    def vector_between(i, j):
        return skeleton[j] - skeleton[i]

    def hinge_quat(parent_vec, child_vec, axis=[0,0,1]):
        """Compute quaternion for hinge joint along a fixed axis"""
        axis = np.array(axis) / np.linalg.norm(axis)
        proj = child_vec - np.dot(child_vec, axis) * axis
        parent_proj = parent_vec - np.dot(parent_vec, axis) * axis
        parent_proj /= np.linalg.norm(parent_proj)
        proj /= np.linalg.norm(proj)
        angle = np.arctan2(np.linalg.norm(np.cross(parent_proj, proj)), np.dot(parent_proj, proj))
        return R.from_rotvec(angle * axis).as_quat()  # x, y, z, w

    def spherical_quat(parent_vec, child_vec):
        """Compute quaternion to rotate parent_vec to child_vec"""
        parent_vec = parent_vec / np.linalg.norm(parent_vec)
        child_vec = child_vec / np.linalg.norm(child_vec)
        cross = np.cross(parent_vec, child_vec)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-6:  # aligned
            return np.array([0,0,0,1])
        angle = np.arctan2(norm_cross, np.dot(parent_vec, child_vec))
        axis = cross / norm_cross
        return R.from_rotvec(angle * axis).as_quat()

    # --- Keypoint indices (based on your skeleton output) ---
    # Spine: root->chest
    idx_root = 0
    idx_chest = 1
    # Right leg
    idx_r_hip = 9
    idx_r_knee = 10
    idx_r_ankle = 11
    # Left leg
    idx_l_hip = 12
    idx_l_knee = 13
    idx_l_ankle = 14
    # Right arm
    idx_r_shoulder = 2
    idx_r_elbow = 3
    idx_r_wrist = 4
    # Left arm
    idx_l_shoulder = 5
    idx_l_elbow = 6
    idx_l_wrist = 7

    # --- Compute limb vectors ---
    v_spine = vector_between(idx_root, idx_chest)

    v_r_thigh = vector_between(idx_r_hip, idx_r_knee)
    v_r_calf = vector_between(idx_r_knee, idx_r_ankle)
    
    v_l_thigh = vector_between(idx_l_hip, idx_l_knee)
    v_l_calf = vector_between(idx_l_knee, idx_l_ankle)
    
    v_r_upper_arm = vector_between(idx_r_shoulder, idx_r_elbow)
    v_r_forearm = vector_between(idx_r_elbow, idx_r_wrist)
    
    v_l_upper_arm = vector_between(idx_l_shoulder, idx_l_elbow)
    v_l_forearm = vector_between(idx_l_elbow, idx_l_wrist)

    # --- Compute quaternions ---
    quats = {}
    # Legs
    quats['r_hip'] = spherical_quat(v_spine, v_r_thigh)
    quats['l_hip'] = spherical_quat(v_spine, v_l_thigh)
    quats['r_knee'] = hinge_quat(v_r_thigh, v_r_calf)
    quats['l_knee'] = hinge_quat(v_l_thigh, v_l_calf)
    # Arms
    quats['r_shoulder'] = spherical_quat(v_spine, v_r_upper_arm)
    quats['l_shoulder'] = spherical_quat(v_spine, v_l_upper_arm)
    quats['r_elbow'] = hinge_quat(v_r_upper_arm, v_r_forearm)
    quats['l_elbow'] = hinge_quat(v_l_upper_arm, v_l_forearm)
    # Spine rotation
    quats['spine'] = spherical_quat(np.array([0,1,0]), v_spine)  # assume Y-up

    return quats


