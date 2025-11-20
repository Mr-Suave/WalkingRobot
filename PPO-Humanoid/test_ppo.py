import gymnasium as gym
import numpy as np
import torch
import glfw
import mujoco

from lib.agent_ppo import PPOAgent
from pose_estimation import load_image, preprocess_image, PoseExtractor, body25_to_humanoid_pose

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Humanoid-v5", render_mode="human")

    # --- Pose Estimation Integration ---
    image_path = "img4.jpg"  
    out_path = "out.jpg"
    try:
        image = load_image(image_path)
        preprocessed_image = preprocess_image(image)
        
        pose_extractor = PoseExtractor()
        body25 = pose_extractor.extract_keypoints(preprocessed_image)
        
        if len(body25) > 0:
            pose = body25_to_humanoid_pose(body25)
            pose_extractor.draw_skeleton(image, body25, out_path)
            obs, _ = env.reset()
            qpos = env.unwrapped.data.qpos.copy()
            qvel = env.unwrapped.data.qvel.copy()
            qpos[18:21] = pose[11:14]  # Right arm
            qpos[21:24] = pose[14:17]  # Left arm
                        
            env.unwrapped.set_state(qpos, qvel)
            obs = env.unwrapped._get_obs()

        else:
            print("Could not detect a pose in the image. Using default reset state.")
            obs, _ = env.reset()

    except FileNotFoundError:
        print(f"Image not found at '{image_path}'. Using default reset state.")
        obs, _ = env.reset()
    # --- End Integration ---

    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load("model.pt",  map_location=torch.device('cpu')))
    agent.eval()
    env.render()
    # Access the viewer through the internal renderer object
    viewer = env.unwrapped.mujoco_renderer.viewer
    # Set camera to ID 0 (typically a fixed wide-angle view)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = 0
    # Maximize the viewer window
    glfw.maximize_window(viewer.window)
    done = False
    # Get the body IDs for the shoulders once, for efficiency
    right_shoulder_id = env.unwrapped.model.body("right_upper_arm").id
    left_shoulder_id = env.unwrapped.model.body("left_upper_arm").id

    # --- TUNE THIS VALUE ---
    # This is a scaling factor for the dynamic force. It's multiplied by the
    # agent's hip torque actions to generate the forward force.
    force_scaling_factor = 0.5  # As requested, "half magnitude"
    # ---------------------

    while not done:
        # Render the frame
        env.render()

        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        
        action_np = action.squeeze(0).cpu().numpy()

        torso_id = env.unwrapped.model.body("torso").id
        constant_force = np.array([6, 0.0, 0.0,   0.0, 0.0, 0.0])  # tune the 0.5


        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

    done = False
    while not done:
        # Render the frame
        env.render()
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        env.unwrapped.data.xfrc_applied[torso_id] = constant_force
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    env.close()
