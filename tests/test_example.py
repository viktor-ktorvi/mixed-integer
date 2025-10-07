from grid2op.Observation import BaseObservation

from tests.scenarios import PPC


def test_scenarios(scenario: tuple[BaseObservation, PPC]):
    assert 1 == 2
