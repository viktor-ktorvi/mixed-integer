import numpy as np
import pytest
from grid2op.Agent import DoNothingAgent, RandomAgent

from scripts.minlp import MINLP
from src.power_flow.validate_equations import validate_equations
from tests.utils import make_obs_from_gekko


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
        "case36_one_gen_on_bus_1_and_one_gen_on_bus_2",
        "case118_default",
    ],
)
def test_validate_equations_predetermined_scenarios(request, env_fixture_name: str, tolerance: float) -> None:
    env = request.getfixturevalue(env_fixture_name)
    validate_equations(env, env.current_obs, threshold=tolerance, verbose=True)


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
        "case36_one_gen_on_bus_1_and_one_gen_on_bus_2",
        # "case118_default",
    ],
)
def test_validate_minlp_problem_predetermined_scenarios(request, env_fixture_name: str, tolerance: float) -> None:
    env = request.getfixturevalue(env_fixture_name)
    obs = env.current_obs

    problem = MINLP(env, obs, validation_mode=True)
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
            problem.fix(problem.Vm[i], problem.net.res_bus.vm_pu[busbar1_id])
            problem.fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[busbar1_id]))
        else:
            problem.fix(problem.Vm[i], problem.net.res_bus.vm_pu[i])
            problem.fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[i]))

    for i in range(problem.n_gen):
        problem.fix(problem.Pg[i], problem.net.res_gen.p_mw[i] / problem.baseMVA)
        problem.fix(problem.Qg[i], problem.net.res_gen.q_mvar[i] / problem.baseMVA)

    # Fix binary switching variables
    for i in range(problem.n_gen):
        problem.fix(problem.a_gen[i], obs.gen_bus[i] - 1)

    for i in range(problem.n_load):
        problem.fix(problem.a_load[i], obs.load_bus[i] - 1)

    for i in range(problem.n_line):
        problem.fix(problem.a_or[i], obs.line_or_bus[i] - 1)
        problem.fix(problem.a_ex[i], obs.line_ex_bus[i] - 1)

    problem.m.options.SOLVER = 1  # APOPT (needed for integer vars)
    problem.m.options.IMODE = 3  # steady-state optimization
    problem.m.options.COLDSTART = 0
    problem.m.Minimize(0)  # no objective, just check constraints

    problem.m.solve(disp=True, debug=True)

    for bus_id in problem.debug_Vm_res:
        Vm = problem.Vm[bus_id].value[0]
        theta = problem.theta[bus_id].value[0]

        Vm_res = problem.debug_Vm_res[bus_id]
        theta_res = problem.debug_theta_res[bus_id]

        if bus_id in env.backend._grid.gen.bus:
            assert np.isclose(Vm, Vm_res)

            if any(env.backend._grid.gen[env.backend._grid.gen.bus == bus_id].slack):
                assert np.isclose(theta, theta_res)

    fake_obs = make_obs_from_gekko(problem)
    validate_equations(env, fake_obs, threshold=tolerance, verbose=True)

    # Print intermediate values to verify balance equations
    for bus_id, P_res, Q_res in problem.residuals:
        assert P_res.value[0] < tolerance
        assert Q_res.value[0] < tolerance

    net = env.backend._grid
    rho_calc = np.zeros(len(obs.p_or))
    for line_id in range(len(obs.p_or)):
        bus_id = obs.line_or_to_subid[line_id]  # or end bus

        S_mva = np.sqrt(obs.p_or[line_id]**2 + obs.q_or[line_id]**2)

        vn_kv = net.bus.loc[bus_id].vn_kv
        Vm_pu = net.res_bus.vm_pu[bus_id]

        V_kv = Vm_pu * vn_kv

        I_actual_A = (S_mva * 1e6) / (np.sqrt(3) * V_kv * 1e3)
        I_limit_A = obs.thermal_limit[line_id]

        rho_calc[line_id] = I_actual_A / I_limit_A

    assert np.isclose(rho_calc, obs.rho)

    # for utilization, rho in zip(problem.utilizations, obs.rho):
    #     assert np.isclose(utilization.value[0], rho)

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
