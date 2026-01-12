import numpy as np
from pypower import idx_bus, idx_gen
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from src.power_flow.model import PowerFlowVariables
from src.utils.power_flow import extract_gekko_variables
from tests.scenarios import PPC


def assert_gekko_pf_solution_matches_pypower(gekko_variables: PowerFlowVariables, ppc: PPC):
    # extract values
    theta_gekko, Vm_gekko, Pg_gekko, Qg_gekko = extract_gekko_variables(gekko_variables, ppc)

    # compare with pypower solution
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    solved_ppc, success = runpf(ppc, ppopt)

    if success == 0:
        raise ValueError("PYPOWER power flow didn't converge successfully.")

    theta_pypower = solved_ppc["bus"][:, idx_bus.VA]
    Vm_pypower = solved_ppc["bus"][:, idx_bus.VM]
    Pg_pypower = solved_ppc["gen"][:, idx_gen.PG]
    Qg_pypower = solved_ppc["gen"][:, idx_gen.QG]

    not_nan_bus = ~np.isnan(solved_ppc["bus"][:, idx_bus.VA])
    not_nan_gen = ~np.isnan(solved_ppc["gen"][:, idx_gen.PG])

    # TODO problem kod case36 default
    #  Pg i Qg su generalno kako valja ali nisu na korektnim mestima (nisu na kako-treba generatoru)
    # fmt: off
    assert all(np.isclose(theta_gekko[not_nan_bus], theta_pypower[not_nan_bus])), "Voltage angles in GEKKO don't match PYPOWER"
    assert all(np.isclose(Vm_gekko[not_nan_bus], Vm_pypower[not_nan_bus])), "Voltage magnitudes in GEKKO don't match PYPOWER"
    assert all(np.isclose(Pg_gekko[not_nan_gen], Pg_pypower[not_nan_gen])), "Generator active powers in GEKKO don't match PYPOWER"
    assert all(np.isclose(Qg_gekko[not_nan_gen], Qg_pypower[not_nan_gen])), "Generator reactive powers in GEKKO don't match PYPOWER"
