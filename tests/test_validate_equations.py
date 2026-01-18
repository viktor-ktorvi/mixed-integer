import pytest

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
    ],
)
def test_validate_equations(request, env_fixture_name):
    env = request.getfixturevalue(env_fixture_name)
    validate_equations(env, env.current_obs, threshold=1e-4, verbose=False)
