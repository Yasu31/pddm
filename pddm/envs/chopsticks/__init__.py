from gym.envs.registration import register

register(
    id='pddm_chopsticks-v0',
    entry_point='pddm.envs.chopsticks.chopsticks_env:ChopsticksEnv',
    max_episode_steps=500,
)
