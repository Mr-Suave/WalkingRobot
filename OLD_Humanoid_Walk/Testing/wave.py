import numpy as np
from humanoid_library import AdamLiteEnv
import time

SCENE_PATH = r"C:\Anvay\Github\WalkingRobot\humanoid_library\humanoid_sim\scene.xml"

env = AdamLiteEnv(scene_xml_path=SCENE_PATH, render_mode="human")
obs, _ = env.reset()

def get_walking_action(t):
    """
    Returns an action array for the humanoid at time t with corrected joint indices.
    """
    action = np.zeros(25)

    # --- Parameters ---
    hip_swing_amp = 0.5       # Hip forward/backward swing (rad)
    knee_bend_amp = 0.7       # Knee bend amplitude (rad)
    hip_shift_amp = 0.15      # Hip side-to-side roll (rad)
    ankle_amp = 0.2           # Ankle pitch for foot lift (rad)
    arm_swing_amp = 0.6       # Arm swing (rad)
    step_freq = 2.5           # Step frequency

    # Phase offsets
    phase = t * step_freq
    left_phase = phase
    right_phase = phase + np.pi

    # -----------------------
    # --- LEG CONTROL (Corrected Indices) ---
    # Hip Pitch (forward/backward)
    action[0] = hip_swing_amp * np.sin(left_phase)       # Left Hip Pitch
    action[6] = hip_swing_amp * np.sin(right_phase)      # Right Hip Pitch

    # Knee Pitch (lift leg)
    action[3] = knee_bend_amp * np.maximum(0, np.sin(left_phase + np.pi/2))   # Left Knee
    action[9] = knee_bend_amp * np.maximum(0, np.sin(right_phase + np.pi/2)) # Right Knee

    # Ankle Pitch (foot lift for step)
    action[4] = -ankle_amp * np.maximum(0, np.sin(left_phase + np.pi/2))      # Left Ankle Pitch
    action[10] = -ankle_amp * np.maximum(0, np.sin(right_phase + np.pi/2))    # Right Ankle Pitch

    # Hip Roll (balance side-to-side)
    action[1] = hip_shift_amp * np.sin(phase)      # Left Hip Roll
    action[7] = -hip_shift_amp * np.sin(phase)     # Right Hip Roll

    # -----------------------
    # --- ARM CONTROL ---
    # Arms swing opposite to legs
    action[15] = arm_swing_amp * np.sin(right_phase)  # Left Shoulder Pitch
    action[20] = arm_swing_amp * np.sin(left_phase)   # Right Shoulder Pitch

    return action

try:
    for i in range(3000):
        t = i * env.dt
        action = get_walking_action(t)  # Use the gait function

        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 200 == 0:
            pelvis_z = obs[2]  # z-position of pelvis (height)
            print(f"Step {i}, Pelvis height: {pelvis_z:.3f}, Waist rotation: {obs[14]:.3f}")
        
        if terminated:
            print("Robot fell! Resetting...")
            obs, _ = env.reset()
            
finally:
    env.close()