from policy import PolicyNetwork
from env import TaskEnvironment
from utils import evaluate_architecture, update_state_with_action
import torch
import torch.optim as optim

# Set up the environment and policy network
task_env = TaskEnvironment(dataset='CIFAR-10', task_type='classification')
policy_network = PolicyNetwork(architecture_size=100, action_space=10)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

def run_nas(num_epochs=1000):
    for epoch in range(num_epochs):
        architecture, log_probs = policy_network.sample_architecture()
        reward = evaluate_architecture(architecture, task_env)  # Get reward by training and testing architecture
        loss = -torch.sum(log_probs) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Reward: {reward}')

if __name__ == "__main__":
    run_nas()