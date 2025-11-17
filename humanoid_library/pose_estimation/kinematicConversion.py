import numpy as np
import cv2
from typing import Tuple, Optional

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

    if not boxes:
        print("No person detected")
        return None

    # Sort by area (largest first)
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    print(f"Detected {len(boxes)} person(s). Trying each until pose found...")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        person_crop = image[y1:y2, x1:x2]
        skeleton = extractor.extract_keypoints(person_crop)

        if skeleton is None or len(skeleton) == 0:
            print(f"Person {i+1}: No pose detected, trying next...")
            continue

        print(f"Pose found for person {i+1}")
        # Map coords back to original image
        skeleton[:, 0] = (x1 + skeleton[:, 0] * (x2 - x1)) / image.shape[1]
        skeleton[:, 1] = (y1 + skeleton[:, 1] * (y2 - y1)) / image.shape[0]

        extractor.draw_skeleton(image, skeleton, save_path)
        return skeleton

    print("No valid skeleton found in any person box")
    return None


def compute_joint_angles_improved(skeleton: np.ndarray) -> np.ndarray:
    """
    Compute 9 joint angles from 2D skeleton that properly map to humanoid_symmetric.xml
    
    Body25 indices (as defined in keyPointExtraction.py):
    0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist
    5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip
    9: RHip, 10: RKnee, 11: RAnkle
    12: LHip, 13: LKnee, 14: LAnkle
    15: REye, 16: LEye, 17: REar, 18: LEar
    19-24: Foot points
    
    Returns 9 angles for:
    [right_hip_y, right_knee, left_hip_y, left_knee,
     right_shoulder1, right_elbow, left_shoulder1, left_elbow, abdomen_y]
    """
    
    # Flip Y-axis: in images, Y increases downward; in 3D, Y increases upward
    skeleton_3d = skeleton.copy()
    skeleton_3d[:, 1] = 1.0 - skeleton_3d[:, 1]
    
    def get_point(idx: int) -> np.ndarray:
        """Get 2D point coordinates"""
        return skeleton_3d[idx, :2]
    
    def compute_angle_from_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute angle of vector (p1->p2) from vertical (downward)
        Returns angle in radians
        - 0 means pointing straight down
        - positive means rotated forward (toward positive X in image = positive Z in 3D)
        - negative means rotated backward
        """
        vec = p2 - p1
        # Vertical down is (0, -1) in our flipped coordinates
        vertical_down = np.array([0, -1])
        
        # Angle from vertical
        angle = np.arctan2(vec[0], -vec[1])  # atan2(x, -y) for angle from down
        return angle
    
    def compute_joint_angle(p_proximal: np.ndarray, p_joint: np.ndarray, 
                           p_distal: np.ndarray) -> float:
        """
        Compute the angle at a joint (like knee or elbow)
        Returns the bending angle in radians
        - 0 means straight
        - negative means bent
        """
        v1 = p_joint - p_proximal  # vector from proximal to joint
        v2 = p_distal - p_joint     # vector from joint to distal
        
        # Angle between the two vectors
        dot_product = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Return as bend angle (negative means bent)
        # In MJCF, knee range is -160 to -2 (negative = bent)
        return -(np.pi - angle)
    
    # Get key points
    neck = get_point(1)
    mid_hip = get_point(8)
    
    # Right leg
    r_hip_pt = get_point(9)
    r_knee_pt = get_point(10)
    r_ankle_pt = get_point(11)
    
    # Left leg
    l_hip_pt = get_point(12)
    l_knee_pt = get_point(13)
    l_ankle_pt = get_point(14)
    
    # Right arm
    r_shoulder_pt = get_point(2)
    r_elbow_pt = get_point(3)
    r_wrist_pt = get_point(4)
    
    # Left arm
    l_shoulder_pt = get_point(5)
    l_elbow_pt = get_point(6)
    l_wrist_pt = get_point(7)
    
    # ===== COMPUTE ANGLES =====
    
    # 1. Right Hip Y (forward/back leg swing)
    # Angle of thigh from vertical down
    right_hip_y = compute_angle_from_vertical(r_hip_pt, r_knee_pt)
    
    # 2. Right Knee (bending angle)
    right_knee = compute_joint_angle(r_hip_pt, r_knee_pt, r_ankle_pt)
    
    # 3. Left Hip Y (forward/back leg swing)
    left_hip_y = compute_angle_from_vertical(l_hip_pt, l_knee_pt)
    
    # 4. Left Knee (bending angle)
    left_knee = compute_joint_angle(l_hip_pt, l_knee_pt, l_ankle_pt)
    
    # 5. Right Shoulder (arm swing)
    # Angle of upper arm from vertical down
    right_shoulder1 = compute_angle_from_vertical(r_shoulder_pt, r_elbow_pt)
    
    # 6. Right Elbow (bending angle)
    right_elbow = compute_joint_angle(r_shoulder_pt, r_elbow_pt, r_wrist_pt)
    
    # 7. Left Shoulder (arm swing)
    left_shoulder1 = compute_angle_from_vertical(l_shoulder_pt, l_elbow_pt)
    
    # 8. Left Elbow (bending angle)
    left_elbow = compute_joint_angle(l_shoulder_pt, l_elbow_pt, l_wrist_pt)
    
    # 9. Spine/Abdomen Y (torso lean)
    # Angle of spine from vertical up
    spine_vec = neck - mid_hip
    vertical_up = np.array([0, 1])
    abdomen_y = np.arctan2(spine_vec[0], spine_vec[1])
    
    # Compile angles
    angles = np.array([
        right_hip_y,
        right_knee,
        left_hip_y,
        left_knee,
        right_shoulder1,
        right_elbow,
        left_shoulder1,
        left_elbow,
        abdomen_y
    ], dtype=np.float32)
    
    # Apply joint limits from humanoid_symmetric.xml
    joint_limits = {
        0: (-120 * np.pi/180, 20 * np.pi/180),   # right_hip_y
        1: (-160 * np.pi/180, -2 * np.pi/180),   # right_knee
        2: (-120 * np.pi/180, 20 * np.pi/180),   # left_hip_y
        3: (-160 * np.pi/180, -2 * np.pi/180),   # left_knee
        4: (-85 * np.pi/180, 60 * np.pi/180),    # right_shoulder1
        5: (-90 * np.pi/180, 50 * np.pi/180),    # right_elbow
        6: (-60 * np.pi/180, 85 * np.pi/180),    # left_shoulder1
        7: (-90 * np.pi/180, 50 * np.pi/180),    # left_elbow
        8: (-75 * np.pi/180, 30 * np.pi/180),    # abdomen_y
    }
    
    # Clip to joint limits
    for i, (min_angle, max_angle) in joint_limits.items():
        angles[i] = np.clip(angles[i], min_angle, max_angle)
    
    print("Computed Joint Angles (degrees):")
    print(f"  Right Hip Y: {angles[0] * 180/np.pi:.1f}°")
    print(f"  Right Knee: {angles[1] * 180/np.pi:.1f}°")
    print(f"  Left Hip Y: {angles[2] * 180/np.pi:.1f}°")
    print(f"  Left Knee: {angles[3] * 180/np.pi:.1f}°")
    print(f"  Right Shoulder: {angles[4] * 180/np.pi:.1f}°")
    print(f"  Right Elbow: {angles[5] * 180/np.pi:.1f}°")
    print(f"  Left Shoulder: {angles[6] * 180/np.pi:.1f}°")
    print(f"  Left Elbow: {angles[7] * 180/np.pi:.1f}°")
    print(f"  Abdomen Y: {angles[8] * 180/np.pi:.1f}°")
    
    return angles


def visualize_3d_pose_projection(skeleton: np.ndarray, 
                                  joint_angles: np.ndarray,
                                  save_path: Optional[str] = None):
    """
    Visualize the computed joint angles overlaid on the skeleton
    """
    # Create a visualization image
    img_size = 512
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Scale skeleton to image size
    skel_scaled = skeleton.copy()
    skel_scaled[:, 0] *= img_size
    skel_scaled[:, 1] = (1 - skel_scaled[:, 1]) * img_size  # Flip Y
    
    # Draw skeleton
    connections = [
        (1, 8), (8, 9), (9, 10), (10, 11),  # Right leg
        (8, 12), (12, 13), (13, 14),         # Left leg
        (1, 2), (2, 3), (3, 4),              # Right arm
        (1, 5), (5, 6), (6, 7),              # Left arm
        (0, 1)                                # Spine
    ]
    
    for start, end in connections:
        if skeleton[start, 2] > 0.3 and skeleton[end, 2] > 0.3:
            pt1 = tuple(skel_scaled[start, :2].astype(int))
            pt2 = tuple(skel_scaled[end, :2].astype(int))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw joints
    for i, (x, y, conf) in enumerate(skel_scaled):
        if conf > 0.3:
            cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    
    # Add angle text
    angle_names = ['RHip', 'RKnee', 'LHip', 'LKnee', 
                   'RShoulder', 'RElbow', 'LShoulder', 'LElbow', 'Spine']
    
    for i, (name, angle) in enumerate(zip(angle_names, joint_angles)):
        text = f"{name}: {angle * 180/np.pi:.1f}°"
        cv2.putText(img, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Pose visualization saved to: {save_path}")
    
    return img
# import numpy as np
# import cv2
# import mediapipe as mp
# from .keyPointExtraction import PoseExtractor
# from ultralytics import YOLO

# def select_main_skeleton_multiple(extractor, image, save_path):
#     """
#     Detect multiple people, extract skeletons, and choose the first non-empty one
#     (prefers largest bounding boxes first).
#     """

#     from ultralytics import YOLO

#     yolo_model = YOLO("yolov8m.pt")

#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)

#     results = yolo_model.predict(source=image, verbose=False)
#     boxes = []

#     for result in results:
#         if result.boxes is None:
#             continue
#         for box in result.boxes:
#             cls = int(box.cls)
#             if cls == 0:  # 'person'
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 boxes.append((x1, y1, x2, y2))
#             else:
#                 print("class: ", cls)

#     if not boxes:
#         print("No person detected")
#         return None

#     # sort by area (largest first)
#     boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
#     print(f"Detected {len(boxes)} person(s). Trying each until pose found...")

#     for i, (x1, y1, x2, y2) in enumerate(boxes):
#         person_crop = image[y1:y2, x1:x2]
#         skeleton = extractor.extract_keypoints(person_crop)

#         if skeleton is None or len(skeleton) == 0:
#             print(f"Person {i+1}: No pose detected, Trying next...")
#             continue

#         print(f"Pose found for person {i+1}")
#         # map coords back to original image
#         skeleton[:, 0] = (x1 + skeleton[:, 0] * (x2 - x1)) / image.shape[1]
#         skeleton[:, 1] = (y1 + skeleton[:, 1] * (y2 - y1)) / image.shape[0]

#         extractor.draw_skeleton(image, skeleton, save_path)
#         return skeleton

#     print("No valid skeleton found in any person box")
#     return None



#     # skeletons = []
#     # for i in range(iterations):
#     #     skel = extractor.extract_keypoints(image)
#     #     if len(skel) > 0:
#     #         skeletons.append(skel)

#     # if not skeletons:
#     #     return None
    
#     # max_area, main_skel = 0 , None

#     # for skel in skeletons:
#     #     x,y = skel[:,0], skel[:,1]
#     #     area = (x.max()-x.min()) * (y.max() - y.min())
#     #     if area > max_area:
#     #         max_area=area
#     #         main_skel = skel
#     # extractor.draw_skeleton(image,main_skel,save_path)
#     # return main_skel



# def compute_joint_angles_improved(skeleton):
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

# # def compute_joint_angles_improved(skeleton):
# #     """
# #     Compute simple 2D joint angles (in radians) for main joints using atan2
# #     """
# #     # gives the limbs of the body as joints
# #     joint_pairs = [
# #         (9,10), # R Hip -> R Knee
# #         (10,11), # R Knee -> R Ankle
# #         (12,13), # L Hip -> L Knee
# #         (13,14), # L Knee -> L Ankle
# #         (2,3), # R shoulder -> R Elbow
# #         (3,4), # R Elbow -> R Wrist
# #         (5,6), # L shoulder -> L Elbow
# #         (6,7), # L Elbow -> L wrist
# #         (8,1) # MidHip -> Neck (spine)
# #     ]
# #     angles=[] #computed angles are stored here

# #     for node1, node2 in joint_pairs:
# #         node1, node2 = skeleton[node1][:2], skeleton[node2][:2] # this gives the x and y coordinates for two nodes of the limb
# #         vec = node2 - node1 #(x2-x1,y2-y1)
# #         angle = np.arctan2(vec[1],vec[0]) #angle wrt to x axis
# #         # angle =0 vector straight right
# #         # angle = pi/2 vector straight up
# #         # angle = pi vector straight left
# #         # angle = -pi/2 vector straing down

# #         angles.append(angle)

# #     return np.array(angles)

