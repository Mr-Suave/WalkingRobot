"""
Test script to visualize pose from image on humanoid URDF
Requirements:
pip install pybullet numpy opencv-python mediapipe
Usage:
python test_humanoid_pose.py
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import os

# -------------------- Your pipeline --------------------
from humanoid_library import (
    load_image,
    preprocess_image,
    PoseExtractor,
    select_main_skeleton_multiple,
)
from humanoid_library.humanoid_sim.reconstruct_and_init import Reconstructor

# -------------------- Helper functions --------------------
def draw_skeleton_overlay(image, body25, save_path=None):
    pose = PoseExtractor()
    pose.draw_skeleton(image, body25, save_path)

# -------------------- Main Test --------------------
def test_pose_on_humanoid(image_path="img1.jpg", save_skeleton_path="skeleton_output.jpg"):

    print("=" * 70)
    print("TEST: HUMANOID INITIAL POSE FROM IMAGE")
    print("=" * 70)

    # 1. Load + preprocess
    print("\n[1/4] Loading + preprocessing image...")
    image = load_image(image_path)
    image_resized = preprocess_image(image)

    # 2. Extract 2D keypoints (BODY25)
    print("\n[2/4] Extracting BODY25 keypoints...")
    pose_extractor = PoseExtractor()

    body25 = select_main_skeleton_multiple(
    extractor=pose_extractor,
    image=image_resized,
    save_path="skeleton_output.jpg"
)

    # Optional: save skeleton overlay
    draw_skeleton_overlay(image_resized, body25, save_skeleton_path)
    print(f"Skeleton overlay saved to: {save_skeleton_path}")

    # 3. Reconstruct 3D joint positions
    print("\n[3/4] Reconstructing 3D pose...")
    recon = Reconstructor()
    # returns joint_positions_cam (25x3), rvec, tvec, camera_matrix, canonical_points
    joint_positions_cam, (rvec, tvec), cam_matrix, canonical_points = recon.reconstruct_3d(
        body25, image_size=image_resized.shape[1::-1]  # (width, height)
    )

    # 4. Apply pose to humanoid in PyBullet
    print("\n[4/4] Visualizing in PyBullet...")

    urdf_path = "simple_humanoid.urdf"
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    # Load plane + humanoid
    p.loadURDF("plane.urdf")
    humanoid_id = p.loadURDF(urdf_path, [0, 0, 1.0], useFixedBase=False)

    # Reset joints using Reconstructor helper
    joint_angles = recon.apply_pose_to_env_simple(humanoid_id, joint_positions_cam)
    print("✓ Pose applied to humanoid.")

    # Simulate for 8 seconds
    for _ in range(240 * 8):
        p.stepSimulation()
        time.sleep(1 / 240)

    print("✓ Done.")
    p.disconnect()

# -------------------- Run script --------------------
if __name__ == "__main__":
    test_pose_on_humanoid(
        image_path="img1.jpg",
        save_skeleton_path="skeleton_output.jpg"
    )
