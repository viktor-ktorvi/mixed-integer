import grid2op
from grid2op.Agent import RandomAgent, DoNothingAgent

from scripts.minlp import MINLP


def main() -> None:
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)

    max_simulation_steps = 10

    agent = RandomAgent(env.action_space)
    agent.seed(0)
    do_nothing_agent = DoNothingAgent(env.action_space)
    counter = 0

    util_limit = 0.9
    max_util_list = []
    while counter < max_simulation_steps:
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        while not done and counter < max_simulation_steps:
            counter += 1

            max_util = float(obs.rho.max())
            max_util_list.append(max_util)

            if max_util < util_limit:
                print(f"Do nothing, {max_util=:.2f}")
                action = do_nothing_agent.act(obs, reward, done)
            else:
                print(f"Solving MINLP, {max_util=:.2f}")

                for sub_id in range(env.n_sub):
                    problem = MINLP(env, obs, utilization_threshold=util_limit)
                    problem.add_bus_type_constraints()
                    problem.add_power_flow_equations()
                    problem.fix_everything_outside_sub(sub_id)

                    problem.m.options.SOLVER = 1  # APOPT (needed for integer vars)
                    # problem.m.options.IMODE = 3  # steady-state optimization

                    problem.m.Minimize(0)  # no objective, just check constraints

                    try:
                        problem.m.solve(disp=False)

                        # TODO nisam siguran u ove akcije; kao da max util ne pada
                        action_dict = problem.get_action_dict(obs)
                        action = env.action_space(action_dict)
                        print(f"Successfully solved MINLP at {sub_id=}")
                        break
                    except:
                        print(f"Failed to solve MINLP, {sub_id=}")

            obs, reward, done, info = env.step(action)

            if any(obs.topo_vect == -1):
                break


if __name__ == "__main__":
    main()