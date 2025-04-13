import gym
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym
from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes



def collect_observations(env_name, num_steps=30):
    """
    Collects event observations from a Gym-compatible OpenTheChests environment.
    """
    env = gym.make(env_name)
    observed_events = []
    obs = env.reset()

    from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes
    observed_events.append(event_to_dict_from_gym(obs, all_types, all_attributes))

    for step in range(num_steps):
        action = [0] * env.action_space.n  # Dummy action
        obs, reward, done, info = env.step(action)
        observed_events.append(event_to_dict_from_gym(obs, all_types, all_attributes))
        if done:
            obs = env.reset()

    env.close()
    return observed_events