from grid2op.Observation import BaseObservation

from src.power_flow.model import get_gekko_power_flow_model
from src.utils.assertions import assert_gekko_pf_solution_matches_pypower
from tests.scenarios import PPC


def test_pf_no_topological_actions(scenario: tuple[BaseObservation, PPC]):
    _, ppc = scenario
    m, variables = get_gekko_power_flow_model(ppc)

    m.options.SOLVER = 1
    m.options.RTOL = 1e-8

    m.solve(disp=True)

    assert_gekko_pf_solution_matches_pypower(variables, ppc)
