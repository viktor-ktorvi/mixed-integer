import copy
from enum import IntEnum

import numpy as np
import pandapower.networks as pn
from gekko import GEKKO
from pandapower import runpp
from pypower import idx_bus, idx_gen
from pypower.makeYbus import makeYbus
from pypower.ppoption import ppoption
from pypower.runpf import runpf


class BusType(IntEnum):
    PQ = 1
    PV = 2
    REF = 3
    ISOLATED = 4


def main():
    # load power grid data
    net = pn.case14()
    # net = pn.case30()
    # net = pn.case57()
    # net = pn.case118()
    # net = pn.case300() # @error: Max Equation Length
    runpp(net)

    if not net.converged:
        raise ValueError("Pandapower power flow did not converge.")

    ppc = copy.deepcopy(net._ppc)

    # define variables
    m = GEKKO(remote=False)

    # define variables
    nb = ppc["bus"].shape[0]

    Vm = m.Array(m.Var, nb, lb=0, value=1)
    theta = m.Array(m.Var, nb, lb=-np.pi, ub=np.pi)

    Pg = m.Array(m.Var, nb)
    Qg = m.Array(m.Var, nb)

    gen_bus_indices = ppc["gen"][:, idx_gen.GEN_BUS].astype(int)
    bus_types = ppc["bus"][:, idx_bus.BUS_TYPE].astype(int)

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

    m.options.SOLVER = 1
    m.options.RTOL = 1e-8
    m.solve(disp=True)

    theta_gekko = np.rad2deg(np.array([theta[i].value[0] for i in range(nb)]))
    Vm_gekko = np.array([Vm[i].value[0] for i in range(nb)])
    Pg_gekko = np.array([Pg[int(i)].value[0] for i in gen_bus_indices]) * ppc["baseMVA"]
    Qg_gekko = np.array([Qg[int(i)].value[0] for i in gen_bus_indices]) * ppc["baseMVA"]

    # compare with pypower solution
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    solved_ppc, success = runpf(ppc, ppopt)

    if success == 0:
        raise ValueError("PYPOWER power flow didn't converge successfully.")

    # fmt: off
    assert all(np.isclose(theta_gekko, solved_ppc["bus"][:, idx_bus.VA])), "Voltage angles in GEKKO don't match PYPOWER"
    assert all(np.isclose(Vm_gekko, solved_ppc["bus"][:, idx_bus.VM])), "Voltage magnitudes in GEKKO don't match PYPOWER"
    assert all(np.isclose(Pg_gekko, solved_ppc["gen"][:, idx_gen.PG])), "Generator active powers in GEKKO don't match PYPOWER"
    assert all(np.isclose(Qg_gekko, solved_ppc["gen"][:, idx_gen.QG])), "Generator reactive powers in GEKKO don't match PYPOWER"
    # fmt: on


if __name__ == "__main__":
    main()
