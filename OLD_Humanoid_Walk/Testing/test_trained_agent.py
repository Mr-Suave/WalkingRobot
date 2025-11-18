import os
import torch
from training.ppo_agent import PPOAgent
from humanoid_library import AdamLiteEnv
import time

def test():
    ################# Hyperparameters #################
    # Make sure the path to scene.xml is correct
    scene_path = os.path.join(os.path.dirname(__file__), '..', 'humanoid_library', 'humanoid_sim', 'scene.xml')
    env = AdamLiteEnv(scene_xml_path=scene_path, render_mode="human") 

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # These parameters are for agent initialization and are not used for updates during testing
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    action_std_init = 0.1 # Use a small, deterministic-like std for testing
    #####################################################

    agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init)

    # Path to the trained model
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'training', "PPO_AdamLite.pth")
    
    # Load the trained model
    try:
        agent.load(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Trained model not found at {checkpoint_path}")
        print("Please run the training script first (training/train.py)")
        env.close()
        return

    # --- Run the simulation ---
    for episode in range(1, 6):
        state, _ = env.reset()
        total_reward = 0
        for t in range(1, 5001):
            action = agent.select_action(state)
            # Scale action to environment's action space
            scaled_action = action * env.action_space.high
            state, reward, terminated, truncated, _ = env.step(scaled_action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"Episode {episode} 	 Timesteps: {t} 	 Total Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == '__main__':
    # Note: You must install PyTorch for this to run: pip install torch
    test()
