import pytest
from grid2op.Observation import BaseObservation

from tests.scenarios import (
    PPC,
    case14_2_lines_and_load_on_busbar_2,
    case14_default,
    case14_line_on_bus_2_on_both_ends,
    case14_line_on_isolated_bus,
    case14_one_gen_on_bus_1_and_one_gen_on_bus_2,
    case14_overloaded,
    case14_substation_with_everything_on_bus_2,
    case36_default,
    case36_one_load_on_bus_2_others_on_bus_1,
    case36_parallel_lines_one_connecting_to_bus_2,
)
from tests.utils import get_scenarios_and_names


def scenarios_and_names() -> tuple[list[tuple[BaseObservation, PPC]], list[str]]:
    scenario_funcs = [
        case14_default,
        case14_2_lines_and_load_on_busbar_2,
        case14_line_on_bus_2_on_both_ends,
        case14_line_on_isolated_bus,
        case14_one_gen_on_bus_1_and_one_gen_on_bus_2,
        case14_substation_with_everything_on_bus_2,
        case14_overloaded,
        case36_default,
        case36_one_load_on_bus_2_others_on_bus_1,
        case36_parallel_lines_one_connecting_to_bus_2,
    ]

    return get_scenarios_and_names(scenario_funcs)


_scenarios, _names = scenarios_and_names()


@pytest.fixture(params=_scenarios, ids=_names)
def scenario(request):
    return request.param
