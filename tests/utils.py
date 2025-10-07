from typing import Callable

from grid2op.Observation import BaseObservation

from tests.scenarios import PPC


def get_scenarios_and_names(
    scenario_funcs: list[Callable[[], tuple[BaseObservation, PPC]]]
) -> tuple[list[tuple[BaseObservation, PPC]], list[str]]:
    scenarios = [scenario_func() for scenario_func in scenario_funcs]
    names = [scenario_func.__name__ for scenario_func in scenario_funcs]
    return scenarios, names
