import os
import torch
from training.ppo_agent import PPOAgent
from humanoid_library import AdamLiteEnv
import numpy as np
import time

def train():
    ################# Hyperparameters #################
    env_name = "AdamLite"
    # The AdamLiteEnv is not registered with Gymnasium, so we instantiate it directly
    # Make sure the path to scene.xml is correct
    scene_path = os.path.join(os.path.dirname(__file__), '..', 'humanoid_library', 'humanoid_sim', 'scene.xml')
    # Use a headless render mode for training
    env = AdamLiteEnv(scene_xml_path=scene_path, render_mode=None) 

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4         # print avg reward in the interval
    log_freq = max_ep_len * 2           # log avg reward in the interval
    save_model_freq = int(1e5)          # save model frequency

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std
    min_action_std = 0.1                # minimum action_std
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency

    update_timestep = max_ep_len * 4    # update policy every n timesteps
    K_epochs = 80                       # update policy for K epochs
    eps_clip = 0.2                      # clip parameter for PPO
    gamma = 0.99                        # discount factor

    lr_actor = 0.0003                   # learning rate for actor network
    lr_critic = 0.001                   # learning rate for critic network
    #####################################################

    print(f"Training on {env_name}")
    print(f"State space dimension: {state_dim}")
    print(f"Action space dimension: {action_dim}")

    agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    #- Path to save trained models
    checkpoint_path = os.path.join(os.path.dirname(__file__), "PPO_AdamLite.pth")
    
    # printing and logging variables
    time_step = 0
    i_episode = 0
    
    start_time = time.time()

    # training loop
    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # Select action with policy
            action = agent.select_action(state)
            # Scale action to environment's action space
            scaled_action = action * env.action_space.high

            state, _, terminated, truncated, info = env.step(scaled_action)

            # --- Custom Reward Function ---
            pelvis_height = state[2]
            forward_velocity = state[env.model.nq] # x-velocity of floating joint

            # 1. Reward for forward velocity
            reward_forward = 2.0 * forward_velocity
            # 2. Reward for staying alive and maintaining height
            reward_alive = 1.0 if pelvis_height > 0.8 else 0.0
            # 3. Penalty for falling
            penalty_fall = -100.0 if pelvis_height <= 0.8 else 0.0
            # 4. Penalty for control effort
            penalty_action = -0.01 * np.sum(np.square(action))

            reward = reward_forward + reward_alive + penalty_action

            done = terminated or truncated or (pelvis_height <= 0.8)
            if done:
                reward += penalty_fall

            # Saving reward and is_terminals:
            # The PPO agent's buffer is being populated by select_action.
            # We now add the reward and done flag to the last entry.
            agent.buffer[-1] = agent.buffer[-1] + (reward, done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()

            # decay action std
            if time_step % action_std_decay_freq == 0:
                new_action_std = agent.action_std - action_std_decay_rate
                new_action_std = round(new_action_std, 4)
                if (new_action_std <= min_action_std):
                    new_action_std = min_action_std
                agent.set_action_std(new_action_std)
                print(f"Action std decayed to {new_action_std}")

            # save model
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print(f"Saving model at timestep {time_step}")
                agent.save(checkpoint_path)
                print(f"Model saved at {checkpoint_path}")
                print("--------------------------------------------------------------------------------------------")

            if done:
                break
        
        print(f"Episode {i_episode} 	 Timesteps: {time_step} 	 Reward: {current_ep_reward:.2f}")
        i_episode += 1

    env.close()
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    # Note: You must install PyTorch for this to run: pip install torch
    train()
