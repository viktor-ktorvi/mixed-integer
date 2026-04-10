from types import SimpleNamespace

import numpy as np
import pytest
from grid2op.Agent import DoNothingAgent, RandomAgent

from scripts.minlp import MINLP, fix
from src.power_flow.validate_equations import validate_equations


@pytest.mark.parametrize(
    "env_fixture_name",
    [
        "case14_default",
        "case14_2_lines_and_load_on_busbar_2",
        "case14_line_on_bus_2_on_both_ends",
        "case14_line_on_isolated_bus",
        "case14_one_gen_on_bus_1_and_one_gen_on_bus_2",
        "case14_substation_with_everything_on_bus_2",
        "case14_overloaded",
        "case36_default",
        "case36_one_load_on_bus_2_others_on_bus_1",
        "case36_parallel_lines_one_connecting_to_bus_2",
        "case36_one_gen_on_bus_1_and_one_gen_on_bus_2" "case118_default",
    ],
)
def test_validate_equations_predetermined_scenarios(request, env_fixture_name: str, tolerance: float) -> None:
    env = request.getfixturevalue(env_fixture_name)
    validate_equations(env, env.current_obs, threshold=tolerance, verbose=True)


# TODO can probably be removed
def make_obs_from_gekko(problem: MINLP) -> SimpleNamespace:
    baseMVA = problem.baseMVA

    return SimpleNamespace(
        gen_p=np.array([problem.Pg[i].value[0] for i in range(problem.n_gen)]) * baseMVA,
        gen_q=np.array([problem.Qg[i].value[0] for i in range(problem.n_gen)]) * baseMVA,
        gen_bus=np.array([int(problem.a_gen[i].value[0]) + 1 for i in range(problem.n_gen)]),
        load_bus=np.array([int(problem.a_load[i].value[0]) + 1 for i in range(problem.n_load)]),
        line_or_bus=np.array([int(problem.a_or[i].value[0]) + 1 for i in range(problem.n_line)]),
        line_ex_bus=np.array([int(problem.a_ex[i].value[0]) + 1 for i in range(problem.n_line)]),
        load_p=problem.obs.load_p,
        load_q=problem.obs.load_q,
        p_or=problem.obs.p_or,
        q_or=problem.obs.q_or,
        p_ex=problem.obs.p_ex,
        q_ex=problem.obs.q_ex,
        line_or_to_subid=problem.obs.line_or_to_subid,
        line_ex_to_subid=problem.obs.line_ex_to_subid,
        thermal_limit=problem.obs.thermal_limit,
    )


@pytest.mark.parametrize(
    "env_fixture_name",
    [
        "case14_default",
        "case14_2_lines_and_load_on_busbar_2",
        "case14_line_on_bus_2_on_both_ends",
        "case14_line_on_isolated_bus",
        "case14_one_gen_on_bus_1_and_one_gen_on_bus_2",
        "case14_substation_with_everything_on_bus_2",
        "case14_overloaded",
        "case36_default",
        "case36_one_load_on_bus_2_others_on_bus_1",
        "case36_parallel_lines_one_connecting_to_bus_2",
        "case36_one_gen_on_bus_1_and_one_gen_on_bus_2"
        # "case118_default",
    ],
)
def test_validate_minlp_problem_predetermined_scenarios(request, env_fixture_name: str, tolerance: float) -> None:
    env = request.getfixturevalue(env_fixture_name)
    obs = env.current_obs

    problem = MINLP(env, obs)
    problem.add_bus_type_constraints()
    problem.add_power_flow_equations()

    # Fix voltage variables
    for i in range(problem.n_bus):
        if np.isnan(problem.net.res_bus.vm_pu[i]):
            # Mirror from the corresponding busbar 1
            if i >= problem.n_sub:
                busbar1_id = i - problem.n_sub  # since busbar 2 = busbar 1 index + n_sub
            else:
                busbar1_id = i + problem.n_sub
            fix(problem.Vm[i], problem.net.res_bus.vm_pu[busbar1_id])
            fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[busbar1_id]))
        else:
            fix(problem.Vm[i], problem.net.res_bus.vm_pu[i])
            fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[i]))

    for i in range(problem.n_gen):
        fix(problem.Pg[i], problem.net.res_gen.p_mw[i] / problem.baseMVA)
        fix(problem.Qg[i], problem.net.res_gen.q_mvar[i] / problem.baseMVA)

    # Fix binary switching variables
    for i in range(problem.n_gen):
        fix(problem.a_gen[i], obs.gen_bus[i] - 1)

    for i in range(problem.n_load):
        fix(problem.a_load[i], obs.load_bus[i] - 1)

    for i in range(problem.n_line):
        fix(problem.a_or[i], obs.line_or_bus[i] - 1)
        fix(problem.a_ex[i], obs.line_ex_bus[i] - 1)

    problem.m.options.SOLVER = 1  # APOPT (needed for integer vars)
    problem.m.options.IMODE = 3  # steady-state optimization
    problem.m.options.COLDSTART = 0
    problem.m.Minimize(0)  # no objective, just check constraints

    try:
        problem.m.solve(disp=True, debug=True)
    except:
        print(f"{problem.m.path=}")

    # TODO 80% sure the power flow equations are good, and that the issue is in the bus types
    fake_obs = make_obs_from_gekko(problem)
    validate_equations(env, fake_obs, threshold=tolerance, verbose=True)

    # Print intermediate values to verify balance equations
    for bus_id, P_res, Q_res in problem.residuals:
        assert P_res.value[0] < tolerance
        assert Q_res.value[0] < tolerance


@pytest.mark.parametrize(
    "env_fixture_name",
    [
        "case14_default",
        "case36_default",
        "case118_default",
    ],
)
def test_validate_equations_random_actions(
    request, env_fixture_name: str, tolerance: float, random_seed: int, max_simulation_steps: int
) -> None:
    env = request.getfixturevalue(env_fixture_name)
    agent = RandomAgent(env.action_space)
    agent.seed(random_seed)
    do_nothing_agent = DoNothingAgent(env.action_space)
    counter = 0
    while counter < max_simulation_steps:
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        while not done and counter < max_simulation_steps:
            counter += 1

            if obs.rho.max() < 0.90:
                action = do_nothing_agent.act(obs, reward, done)
            else:
                action = agent.act(obs, reward, done)

            obs, reward, done, info = env.step(action)

            if any(obs.topo_vect == -1):
                break

            validate_equations(env, obs, threshold=1e-4, verbose=False)
