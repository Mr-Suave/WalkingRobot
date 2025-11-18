import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

class Actor(nn.Module):
    """
    The Actor network takes a state and outputs a policy (a distribution over actions).
    For continuous control, this is typically the mean and standard deviation of a
    Gaussian distribution.
    """
    def __init__(self, state_dim, action_dim, action_std_init):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        action_mean = self.net(state)
        cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_matrix)
        return dist

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

class Critic(nn.Module):
    """
    The Critic network takes a state and estimates the value of that state (V-value).
    It helps the Actor by providing a baseline to judge whether an action was
    better or worse than expected.
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state)

class PPOAgent:
    """
    The main PPO Agent class. It holds the Actor and Critic networks,
    manages the experience buffer, and implements the PPO update rule.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.action_std = action_std_init

        # The "old" policy is the one we used to collect the latest batch of experience.
        # We need to keep it around to calculate the policy ratio for the PPO objective.
        self.policy_old = Actor(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy_old.state_dict()) # Make a copy

        # The "new" policy is the one we are currently training.
        self.policy = Actor(state_dim, action_dim, action_std_init)
        self.critic = Critic(state_dim)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()
        self.buffer = []

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        """
        Select an action from the policy distribution for a given state.
        This is the "acting" part of the agent.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            dist = self.policy_old(state)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        self.buffer.append((state, action, action_log_prob))
        
        return action.detach().cpu().numpy().flatten()

    def update(self):
        """
        Update the Actor and Critic networks using the collected experience.
        This is the "learning" part of the agent.
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for _, _, _, reward, is_terminal in reversed(self.buffer):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack([item[0] for item in self.buffer], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack([item[1] for item in self.buffer], dim=0)).detach()
        old_log_probs = torch.squeeze(torch.stack([item[2] for item in self.buffer], dim=0)).detach()

        # PPO Update for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer = []

    def evaluate(self, state, action):
        """
        Helper function to evaluate the state-value and log probability of an action
        under the current policy.
        """
        dist = self.policy(state)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_log_probs, torch.squeeze(state_values), dist_entropy

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
