import copy

import numpy as np
import pandapower.networks as pn
from pandapower import runpp
from pypower import idx_bus, idx_gen
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from src.power_flow.model import get_gekko_power_flow_model
from src.power_flow.utils import get_gen_bus_indices


def main():
    # load power grid data
    net = pn.case118()
    runpp(net)

    if not net.converged:
        raise ValueError("Pandapower power flow did not converge.")

    ppc = copy.deepcopy(net._ppc)

    m, variables = get_gekko_power_flow_model(ppc)

    m.options.SOLVER = 1
    m.options.RTOL = 1e-8

    m.solve(disp=True)

    theta = variables.theta
    Vm = variables.Vm
    Pg = variables.Pg
    Qg = variables.Qg

    nb = ppc["bus"].shape[0]

    gen_bus_indices = get_gen_bus_indices(ppc)
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

    # print results

    field_size = 15
    decimals = 3
    print(
        f"{'bus':^{field_size}}| {'theta [rad]':^{field_size}}| {'Vm [p.u.]':^{field_size}}| {'Pg [p.u.]':^{field_size}}| {'Qg [p.u.]':^{field_size}}"
    )
    for i in range(nb):
        print(
            f"{i:^{field_size}}| {theta[i].value[0]:^{field_size}.{decimals}}| {Vm[i].value[0]:^{field_size}.{decimals}}| {Pg[i].value[0]:^{field_size}.{decimals}}| {Qg[i].value[0]:^{field_size}.{decimals}}"
        )


if __name__ == "__main__":
    main()
