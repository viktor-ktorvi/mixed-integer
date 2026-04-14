import warnings

import grid2op
import pandas as pd
from grid2op.Agent import RandomAgent, DoNothingAgent

from scripts.minlp import MINLP


def main() -> None:
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)

    max_simulation_steps = 100000

    agent = RandomAgent(env.action_space)
    agent.seed(0)
    do_nothing_agent = DoNothingAgent(env.action_space)
    counter = 0

    util_limit_max = 1.0
    util_limit_min = 0.9
    util_limit = util_limit_min
    # TODO sa util_limit = 0.8 ima nekih upitnih situacija tipa
    #  Successfully solved MINLP at sub_id=12 minlp_max_util=0.6324
    #  Solving MINLP, counter=382 max_util=1.03
    #  gde se stvari ne podudaraju; ove je za istragu; veoma je sumnjivo
    max_util_list = []
    solving_substation_histogram = {sub_id: 0 for sub_id in range(env.n_sub)}
    while counter < max_simulation_steps:
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        while not done and counter < max_simulation_steps:
            counter += 1

            max_util = float(obs.rho.max())
            max_util_list.append(max_util)

            solved_successfully = False
            if max_util < util_limit:
                print(f"Do nothing, {counter=} {max_util=:.2f}")
                action = do_nothing_agent.act(obs, reward, done)
                solved_successfully = True
            else:
                print(f"Solving MINLP, {counter=} {max_util=:.2f}")

                sorted_solving_substation_histogram = dict(sorted(solving_substation_histogram.items(), key=lambda x: x[1], reverse=True))


                for sub_id in sorted_solving_substation_histogram:
                    problem = MINLP(env, obs, utilization_threshold=util_limit)
                    problem.add_bus_type_constraints()
                    problem.add_power_flow_equations()
                    problem.fix_everything_outside_sub(sub_id)

                    problem.m.options.SOLVER = 1  # APOPT (needed for integer vars)
                    # problem.m.options.IMODE = 3  # steady-state optimization

                    # TODO da li moze da se smanji max iter? da se ubrzaju neuspesna resavanja?

                    problem.m.Minimize(problem.t)

                    try:
                        problem.m.solve(disp=False)
                        solution_max_util = max(float(u.value[0]) for u in problem.utilizations.values())

                        try:
                            action_dict = problem.get_action_dict(obs)
                            action = env.action_space(action_dict)
                        except Exception as e:
                            print(e)
                        print(f"Successfully solved MINLP at {sub_id=} minlp_max_util={solution_max_util:.4f}")
                        solving_substation_histogram[sub_id] += 1
                        solved_successfully = False
                        break
                    except:
                        print(f"Failed to solve MINLP, {sub_id=}")
                        solved_successfully = False

            if not solved_successfully:
                util_limit += 0.05
            else:
                util_limit -= 0.05

            if util_limit > util_limit_max:
                util_limit = util_limit_max
            if util_limit < util_limit_min:
                util_limit = util_limit_min

            obs, reward, done, info = env.step(action)

            if any(obs.topo_vect == -1):
                break


if __name__ == "__main__":
    main()