import torch
import torch.nn as nn
from utils import update_state_with_action


class PolicyNetwork(nn.Module):
    def __init__(self, architecture_size, action_space):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=architecture_size, hidden_size=128)
        self.fc = nn.Linear(128, action_space)  # Outputs probabilities for architecture decisions

    def forward(self, state):
        output, _ = self.lstm(state)
        return self.fc(output)

    def sample_architecture(self):
        architecture, log_probs = [], []
        state = torch.zeros(1, 1, 100)  # Initial state with architecture_size=100
        for _ in range(10):  # For each architectural decision
            probs = torch.softmax(self.forward(state), dim=-1)  # Forward pass and softmax
            probs = probs.squeeze(0)  # Remove any extra dimension
            if probs.dim() > 1:  # Sometimes you get a [1, N] tensor; reduce it to [N]
                probs = probs.squeeze(0)
            action = torch.distributions.Categorical(probs).sample()  # Sample an action
            log_prob = torch.log(probs[action])  # Get the log probability of the chosen action
            architecture.append(action.item())  # Save the chosen action
            log_probs.append(log_prob)  # Save the log probability
            state = update_state_with_action(state, action)  # Update the state for the next decision
        return architecture, torch.stack(log_probs)

