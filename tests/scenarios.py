import grid2op
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation

from src.utils.grid2op import (
    get_empty_action_dict,
    set_gen_buses,
    set_line_buses,
    set_load_buses,
)

PPC = dict


def case14_env(random_seed: int = 0) -> Environment:
    env_name = "l2rpn_case14_sandbox"  # for example, other environments might be usable
    env = grid2op.make(env_name)
    env.seed(random_seed)

    return env


def case36_env(random_seed: int = 0) -> Environment:
    env_name = "l2rpn_icaps_2021_small"  # for example, other environments might be usable
    env = grid2op.make(env_name)
    env.seed(random_seed)

    return env


def case14_default() -> tuple[BaseObservation, PPC]:
    env = case14_env()
    obs = env.reset()
    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_2_lines_and_load_on_busbar_2() -> tuple[BaseObservation, PPC]:
    env = case14_env()
    obs = env.reset()

    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[6, 4], sub_ids=[4, 4], bus_ids=[2, 2], action_dict=action_dict, obs=obs)
    action_dict = set_load_buses(load_ids=[3], bus_ids=[2], action_dict=action_dict)

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_line_on_bus_2_on_both_ends() -> tuple[BaseObservation, PPC]:
    env = case14_env()

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[16, 11], sub_ids=[8, 8], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[14, 11], sub_ids=[13, 13], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action_dict = set_load_buses(
        load_ids=[10],
        bus_ids=[2],
        action_dict=action_dict,
    )
    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_line_on_isolated_bus() -> tuple[BaseObservation, PPC]:
    env = case14_env()

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[6], sub_ids=[4], bus_ids=[2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_one_gen_on_bus_1_and_one_gen_on_bus_2() -> tuple[BaseObservation, PPC]:
    env = case14_env()

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[8], sub_ids=[5], bus_ids=[2], action_dict=action_dict, obs=obs)

    action_dict = set_gen_buses(
        gen_ids=[2],
        bus_ids=[2],
        action_dict=action_dict,
    )
    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_substation_with_everything_on_bus_2() -> tuple[BaseObservation, PPC]:
    env = case14_env()

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
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case14_overloaded() -> tuple[BaseObservation, PPC]:
    env = case14_env()

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[17, 7], sub_ids=[5, 5], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)
    ppc = env.backend._grid._ppc
    return obs, ppc


def case36_default() -> tuple[BaseObservation, PPC]:
    env = case36_env()

    obs = env.reset()
    ppc = env.backend._grid._ppc
    return obs, ppc


def case36_one_load_on_bus_2_others_on_bus_1() -> tuple[BaseObservation, PPC]:
    env = case36_env()

    obs = env.reset()
    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[53], sub_ids=[35], bus_ids=[2], action_dict=action_dict, obs=obs)
    action_dict = set_load_buses(
        load_ids=[35],
        bus_ids=[2],
        action_dict=action_dict,
    )

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)

    ppc = env.backend._grid._ppc
    return obs, ppc


def case36_parallel_lines_one_connecting_to_bus_2() -> tuple[BaseObservation, PPC]:
    env = case36_env()
    obs = env.reset()

    action_dict = get_empty_action_dict()
    action_dict = set_line_buses(line_ids=[18, 49], sub_ids=[16, 16], bus_ids=[2, 2], action_dict=action_dict, obs=obs)

    action = env.action_space(action_dict)
    obs, _, _, _ = env.step(action)
    ppc = env.backend._grid._ppc
    return obs, ppc
