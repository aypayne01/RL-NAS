import ray

ray.init()

@ray.remote
def evaluate_architecture_remote(architecture, task_env):
    return task_env.evaluate(architecture)

def distributed_nas(policy_network, task_env, num_architectures=5):
    architectures, log_probs = policy_network.sample_architecture()
    rewards = ray.get([evaluate_architecture_remote.remote(arch, task_env) for arch in architectures])
    policy_network.update(rewards, log_probs)