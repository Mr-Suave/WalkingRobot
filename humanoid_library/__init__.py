"""
Humanoid Library - Pose Estimation and RL Walking Simulation
"""

# ---- Pose Estimation ----
from .pose_estimation.imgAcquisition import load_image, preprocess_image
from .pose_estimation.keyPointExtraction import PoseExtractor
from .pose_estimation.kinematicConversion import (
    select_main_skeleton_multiple,
    compute_joint_angles_improved,  # This is the OLD function from kinematicConversion.py
)
from .humanoid_sim.visualize import AdamLiteEnv


__all__ = [
    # Pose estimation
    "load_image", 
    "preprocess_image",
    "PoseExtractor", 
    "select_main_skeleton_multiple",
    "compute_joint_angles_improved", # OLD function from kinematicConversion.py
    "AdamLiteEnv"
]

__version__ = "0.2.0"