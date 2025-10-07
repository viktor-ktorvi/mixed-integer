from typing import Callable

from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt

from tests.scenarios import (
    PPC,
    case14_2_lines_and_load_on_busbar_2,
    case14_default,
    case14_env,
    case14_line_on_bus_2_on_both_ends,
    case14_line_on_isolated_bus,
    case14_one_gen_on_bus_1_and_one_gen_on_bus_2,
    case14_overloaded,
    case14_substation_with_everything_on_bus_2,
    case36_default,
    case36_env,
    case36_one_load_on_bus_2_others_on_bus_1,
    case36_parallel_lines_one_connecting_to_bus_2,
)
from tests.utils import get_scenarios_and_names


def plot_scenarios(env: Environment, scenario_funcs: list[Callable[[], tuple[BaseObservation, PPC]]]):
    plot_helper = PlotMatplot(env.observation_space, line_id=True, load_id=True, gen_id=True)
    scenarios, scenario_names = get_scenarios_and_names(scenario_funcs)

    for i in range(len(scenarios)):
        plt.figure()
        plot_helper.plot_obs(scenarios[i][0])
        ax = plt.gca()
        ax.set_title(scenario_names[i])


def main():
    case14_scenario_funcs = [
        case14_default,
        case14_2_lines_and_load_on_busbar_2,
        case14_line_on_bus_2_on_both_ends,
        case14_line_on_isolated_bus,
        case14_one_gen_on_bus_1_and_one_gen_on_bus_2,
        case14_substation_with_everything_on_bus_2,
        case14_overloaded,
    ]
    case36_scenario_funcs = [
        case36_default,
        case36_one_load_on_bus_2_others_on_bus_1,
        case36_parallel_lines_one_connecting_to_bus_2,
    ]

    plot_scenarios(case14_env(), case14_scenario_funcs)
    plot_scenarios(case36_env(), case36_scenario_funcs)

    plt.show()


if __name__ == "__main__":
    main()
