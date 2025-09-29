import copy

import numpy as np
import pandapower.networks as pn
from pandapower import runpp
from pypower import idx_bus, idx_gen
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from src.power_flow.model import get_gekko_power_flow_model


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
