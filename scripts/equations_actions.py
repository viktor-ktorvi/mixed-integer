import time

import grid2op
from grid2op.Agent import DoNothingAgent, RandomAgent, TopologyGreedy

from scripts.equations import validate_equations


def main():
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)

    agent = TopologyGreedy(env.action_space)
    # agent = RandomAgent(env.action_space)
    do_nothing_agent = DoNothingAgent(env.action_space)
    nb_episode = 1
    for _ in range(nb_episode):
        # TODO bilo bi dobro sacuvati obs i net u svakom momentu kao pickle fajlove
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        while not done:
            time_start = time.time()

            if obs.rho.max() < 0.90:
                action = do_nothing_agent.act(obs, reward, done)
            else:
                action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)

            validate_equations(env, obs, threshold=1e-4, verbose=False)

            time_taken = time.time() - time_start
            print(f"{obs.rho.max()=:5.2f} {time_taken=:5.3f}")


if __name__ == "__main__":
    main()
