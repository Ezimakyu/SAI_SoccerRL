import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional

# Re-use the basic NeuralNetwork structure if desired, but we need a specific Gaussian Policy
class GaussianPolicy(nn.Module):
    def __init__(
        self,
        n_features,
        n_actions,
        neurons,
        activation_function=F.relu,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.base_net = nn.Sequential()
        in_dim = n_features
        for i, out_dim in enumerate(neurons):
            self.base_net.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            self.base_net.add_module(f"activation_{i}", nn.ReLU())
            in_dim = out_dim
            
        self.mean_linear = nn.Linear(in_dim, n_actions)
        self.log_std_linear = nn.Linear(in_dim, n_actions)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        
    def forward(self, state):
        # This remains standard for SAC logic
        x = self.base_net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    # NEW: Clean inference method for TorchScript
    @torch.jit.export
    def get_action(self, state):
        """Purely mathematical path for submission/inference."""
        x = self.base_net(state)
        mean = self.mean_linear(x)
        return torch.tanh(mean)

    def sample(self, state):
        # Training logic
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

class QNetwork(nn.Module):
    def __init__(self, n_features, n_actions, neurons, activation_function=F.relu):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential()
        in_dim = n_features + n_actions
        for i, out_dim in enumerate(neurons):
            self.q1.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            self.q1.add_module(f"activation_{i}", nn.ReLU())
            in_dim = out_dim
        self.q1.add_module("output", nn.Linear(in_dim, 1))

        # Q2 architecture
        self.q2 = nn.Sequential()
        in_dim = n_features + n_actions
        for i, out_dim in enumerate(neurons):
            self.q2.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            self.q2.add_module(f"activation_{i}", nn.ReLU())
            in_dim = out_dim
        self.q2.add_module("output", nn.Linear(in_dim, 1))

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)

class SAC(nn.Module):
    def __init__(
        self,
        n_features,
        action_space,
        neurons=[256, 256],
        activation_function=F.relu,
        learning_rate=3e-4,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        alpha: float = 0.2,
        alpha_decay: Optional[float] = None,
        alpha_min: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = float(alpha)
        self.alpha_decay = float(alpha_decay) if alpha_decay is not None else None
        self.alpha_min = float(alpha_min)
        self._update_steps = 0
        self.action_space = action_space

        self.critic = QNetwork(n_features, action_space.shape[0], neurons)
        self.critic_target = QNetwork(n_features, action_space.shape[0], neurons)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = GaussianPolicy(
            n_features,
            action_space.shape[0],
            neurons,
            activation_function=activation_function,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        # Move modules to device (GPU if available)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        self.critic.eval()
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones, epochs=1):
        self.actor.train()
        self.critic.train()

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Optional entropy coefficient schedule (useful for standing/balance)
        if self.alpha_decay is not None:
            # decay once per update() call (not per gradient step inside epochs loop)
            self._update_steps += 1
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

        for _ in range(epochs):
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(next_states)
                qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target

            qf1, qf2 = self.critic(states, actions)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optimizer.zero_grad()
            qf_loss.backward()
            self.critic_optimizer.step()

            pi, log_pi, _ = self.actor.sample(states)
            qf1_pi, qf2_pi = self.critic(states, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf_loss.item(), policy_loss.item()
