"""
Humanoid Library - Pose Estimation and RL Walking Simulation
"""

# ---- Pose Estimation ----
from .pose_estimation.imgAcquisition import load_image, preprocess_image
from .pose_estimation.keyPointExtraction import PoseExtractor
from .pose_estimation.kinematicConversion import (
    select_main_skeleton_multiple,
    compute_joint_angles,  # This is the OLD function from kinematicConversion.py
    vector_to_rotation
)

# ---- Humanoid Simulation (NEW) ----
from .humanoid_sim.skeleton_to_urdf import SkeletonToURDFMapper  # NEW mapper class
from .humanoid_sim.humanoid_walking_env import HumanoidWalkingEnv  # NEW RL env
from .humanoid_sim.reconstruct_and_init import Reconstructor


__all__ = [
    # Pose estimation
    "load_image", 
    "preprocess_image",
    "PoseExtractor", 
    "select_main_skeleton_multiple",
    "compute_joint_angles",  # OLD function from kinematicConversion.py
    "vector_to_rotation",
    
    # NEW: RL components
    "SkeletonToURDFMapper",  # NEW mapper class
    "HumanoidWalkingEnv",     # NEW walking environment
    "Reconstructor"
]

__version__ = "0.2.0"