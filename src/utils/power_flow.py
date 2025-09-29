import numpy as np
from numpy import typing as npt
from pypower import idx_gen

from src.power_flow.model import PowerFlowVariables


def extract_gekko_variables(
    variables: PowerFlowVariables, ppc: dict
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    theta = variables.theta
    Vm = variables.Vm
    Pg = variables.Pg
    Qg = variables.Qg

    # extract values
    nb = ppc["bus"].shape[0]

    theta_gekko = np.rad2deg(np.array([theta[i].value[0] for i in range(nb)]))
    Vm_gekko = np.array([Vm[i].value[0] for i in range(nb)])

    # extract generator values (account for multiple generators )
    gen_bus_indices = list(ppc["gen"][:, idx_gen.GEN_BUS].astype(int))
    ng = ppc["gen"].shape[0]
    Pg_gekko = np.zeros((ng,))
    Qg_gekko = np.zeros((ng,))
    occurrence_flags = {key: False for key in np.unique(gen_bus_indices)}
    for i in range(ng):
        bus_idx = gen_bus_indices[i]
        count = gen_bus_indices.count(bus_idx)

        Pg_gekko[i] = Pg[bus_idx].value[0] * ppc["baseMVA"] * int(not occurrence_flags[bus_idx])
        Qg_gekko[i] = Qg[bus_idx].value[0] * ppc["baseMVA"] / count

        if count > 1:
            occurrence_flags[bus_idx] = True

    return theta_gekko, Vm_gekko, Pg_gekko, Qg_gekko
