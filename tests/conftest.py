import grid2op
import pytest
from grid2op.Environment import Environment

from src.utils.grid2op import (
    get_empty_action_dict,
    set_gen_buses,
    set_line_buses,
    set_load_buses,
)


@pytest.fixture
def random_seed() -> int:
    return 0


@pytest.fixture
def tolerance() -> float:
    return 1e-4


@pytest.fixture
def max_simulation_steps() -> int:
    return 20


@pytest.fixture
def case14_env(random_seed: int) -> Environment:
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)
    env.seed(random_seed)

    return env


@pytest.fixture
def case14_default(case14_env: Environment) -> Environment:
    case14_env.reset()
    return case14_env


@pytest.fixture
def case14_2_lines_and_load_on_busbar_2(case14_default: Environment) -> Environment:
    env = case14_default
    obs = env.reset()

    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[6, 4], sub_ids=[4, 4], bus_ids=[2, 2], action_dict=action_dict, obs=obs)
    action_dict = set_load_buses(load_ids=[3], bus_ids=[2], action_dict=action_dict)

    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case14_line_on_bus_2_on_both_ends(case14_default: Environment) -> Environment:
    env = case14_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[16, 11], sub_ids=[8, 8], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    obs, *_ = env.step(action)
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[14, 11], sub_ids=[13, 13], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action_dict = set_load_buses(
        load_ids=[10],
        bus_ids=[2],
        action_dict=action_dict,
    )
    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case14_line_on_isolated_bus(case14_default: Environment) -> Environment:
    env = case14_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[6], sub_ids=[4], bus_ids=[2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case14_one_gen_on_bus_1_and_one_gen_on_bus_2(case14_default: Environment) -> Environment:
    env = case14_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[8], sub_ids=[5], bus_ids=[2], action_dict=action_dict, obs=obs)

    action_dict = set_gen_buses(
        gen_ids=[2],
        bus_ids=[2],
        action_dict=action_dict,
    )
    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture()
def case14_substation_with_everything_on_bus_2(case14_default: Environment) -> Environment:
    env = case14_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(
        line_ids=[7, 8, 9, 17], sub_ids=[5, 5, 5, 5], bus_ids=[2, 2, 2, 2], action_dict=action_dict, obs=obs
    )
    action_dict = set_gen_buses(
        gen_ids=[2, 3],
        bus_ids=[2, 2],
        action_dict=action_dict,
    )
    action_dict = set_load_buses(
        load_ids=[4],
        bus_ids=[2],
        action_dict=action_dict,
    )
    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case14_overloaded(case14_default: Environment) -> Environment:
    env = case14_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[17, 7], sub_ids=[5, 5], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case36_env(random_seed: int) -> Environment:
    env_name = "l2rpn_icaps_2021_small"
    env = grid2op.make(env_name)
    env.seed(random_seed)

    return env


@pytest.fixture
def case36_default(case36_env: Environment) -> Environment:
    case36_env.reset()
    return case36_env


@pytest.fixture
def case36_one_load_on_bus_2_others_on_bus_1(case36_default: Environment) -> Environment:
    env = case36_default

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[53], sub_ids=[35], bus_ids=[2], action_dict=action_dict, obs=obs)
    action_dict = set_load_buses(
        load_ids=[35],
        bus_ids=[2],
        action_dict=action_dict,
    )

    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case36_parallel_lines_one_connecting_to_bus_2(case36_default: Environment) -> Environment:
    env = case36_default
    obs = env.reset()

    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[18, 49], sub_ids=[16, 16], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    env.step(action)

    return env


@pytest.fixture
def case118_env(random_seed: int) -> Environment:
    env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)
    env.seed(random_seed)

    return env


@pytest.fixture
def case118_default(case118_env: Environment) -> Environment:
    case118_env.reset()
    return case118_env
