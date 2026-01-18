import pytest
from grid2op.Agent import DoNothingAgent, RandomAgent

from scripts.equations import validate_equations


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
        "case118_default",
    ],
)
def test_validate_equations_predetermined_scenarios(request, env_fixture_name: str, tolerance: float) -> None:
    env = request.getfixturevalue(env_fixture_name)
    validate_equations(env, env.current_obs, threshold=tolerance, verbose=False)


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
