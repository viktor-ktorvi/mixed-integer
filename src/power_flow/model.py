from dataclasses import dataclass

import numpy as np
from gekko import GEKKO
from gekko.gk_variable import GKVariable
from pypower import idx_bus, idx_gen
from pypower.makeYbus import makeYbus

from src.power_flow.utils import BusType, get_bus_types, get_gen_bus_indices


@dataclass
class PowerFlowVariables:
    theta: GKVariable
    Vm: GKVariable
    Pg: GKVariable
    Qg: GKVariable


def get_gekko_power_flow_model(ppc: dict) -> tuple[GEKKO, PowerFlowVariables]:
    """
    Construct a GEKKO model for the AC power flow equations.

    Parameters
    ----------
    ppc: dict

    Returns
    -------
    m: GEKKO
    variables: PowerFlowVariables
    """
    m = GEKKO(remote=False)

    # define variables
    nb = ppc["bus"].shape[0]

    Vm = m.Array(m.Var, nb, lb=0, value=1)
    theta = m.Array(m.Var, nb, lb=-np.pi, ub=np.pi)

    Pg = m.Array(m.Var, nb)
    Qg = m.Array(m.Var, nb)

    gen_bus_indices = get_gen_bus_indices(ppc)
    bus_types = get_bus_types(ppc)

    # fix variables that are actually constant
    for i in range(nb):
        if bus_types[i] == BusType.REF:
            m.fix(Vm[i], val=ppc["bus"][i, idx_bus.VM])
            m.fix(theta[i], val=np.deg2rad(ppc["bus"][i, idx_bus.VA]))

        if bus_types[i] == BusType.PQ:
            if i in gen_bus_indices:
                gen_idx = np.argwhere(gen_bus_indices == i).item()
                m.fix(Pg[i], val=ppc["gen"][gen_idx, idx_gen.PG] / ppc["baseMVA"])
                m.fix(Qg[i], val=ppc["gen"][gen_idx, idx_gen.QG] / ppc["baseMVA"])
            else:
                m.fix(Pg[i], val=0)
                m.fix(Qg[i], val=0)

        if bus_types[i] == BusType.PV:
            m.fix(Vm[i], val=ppc["bus"][i, idx_bus.VM])
            if i in gen_bus_indices:
                gen_idx = np.argwhere(gen_bus_indices == i).item()
                m.fix(Pg[i], val=ppc["gen"][gen_idx, idx_gen.PG] / ppc["baseMVA"])
            else:
                m.fix(Pg[i], val=0)

    # add parameters
    Pd = ppc["bus"][:, idx_bus.PD] / ppc["baseMVA"]
    Qd = ppc["bus"][:, idx_bus.QD] / ppc["baseMVA"]

    P = Pg - Pd
    Q = Qg - Qd

    Ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    Gbus = Ybus.real.toarray()
    Bbus = Ybus.imag.toarray()

    # fmt: off
    # active power conservation
    m.Equations(
        [
            0 == -P[i] + sum([Vm[i] * Vm[k] * (Gbus[i, k] * m.cos(theta[i] - theta[k]) + Bbus[i, k] * m.sin(theta[i] - theta[k])) for k in range(nb)]) for i in range(nb)
        ]
    )

    # reactive power conservation
    m.Equations(
        [
            0 == -Q[i] + sum([Vm[i] * Vm[k] * (Gbus[i, k] * m.sin(theta[i] - theta[k]) - Bbus[i, k] * m.cos(theta[i] - theta[k])) for k in range(nb)]) for i in range(nb)
        ]
    )
    # fmt: on

    variables = PowerFlowVariables(theta=theta, Vm=Vm, Pg=Pg, Qg=Qg)

    return m, variables
