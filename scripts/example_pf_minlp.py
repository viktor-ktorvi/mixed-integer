import copy
import random

import grid2op
import numpy as np
from gekko import GEKKO
from grid2op.Agent import RandomAgent
from grid2op.Observation import BaseObservation
from pypower import idx_brch, idx_bus, idx_gen
from pypower.makeYbus import makeYbus
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from src.power_flow.model import PowerFlowVariables
from src.utils.power_flow import extract_gekko_variables
from src.utils.pypower import BusType, get_bus_types, get_gen_bus_indices


def verify_equations(ppc: dict, obs: BaseObservation):
    # calculate pypower current
    Vm_pu = ppc["bus"][:, idx_bus.VM]

    V_pu = Vm_pu * np.exp(1j * np.deg2rad(ppc["bus"][:, idx_bus.VA]))

    Ybus, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    If_pu = Yf @ V_pu

    idx_f = ppc["branch"][:, idx_brch.F_BUS].astype(int)
    base_kV_f = ppc["bus"][idx_f, idx_bus.BASE_KV]

    base_If_A = ppc["baseMVA"] / base_kV_f * 1000

    If_phase_A_pypower = np.abs(If_pu) * base_If_A / np.sqrt(3)

    a_or = obs.a_or

    assert all(np.isclose(If_phase_A_pypower, a_or))

    # define problem
    m = GEKKO(remote=False)

    # define variables
    nb = ppc["bus"].shape[0]
    nl = ppc["branch"].shape[0]

    n_sub = nb // 2
    assert nb % 2 == 0

    Vm = m.Array(m.Var, nb, lb=0, value=1)
    theta = m.Array(m.Var, nb, lb=-np.pi, ub=np.pi)

    Pg = m.Array(m.Var, nb)
    Qg = m.Array(m.Var, nb)

    gen_bus_indices = get_gen_bus_indices(ppc)
    bus_types = get_bus_types(ppc)

    gen_actions_currently = obs.gen_bus - 1
    a_gen = gen_actions_currently

    # fix variables that are actually constant
    for i in range(nb):
        gen_selector = 1 - a_gen[gen_bus_indices == i] if i < n_sub else a_gen[gen_bus_indices == i]

        # TODO ne znam da li ce element-wise mnozenje smeti; u suprotnom for petlja u sumo
        active_power_gen = sum(gen_selector * ppc["gen"][gen_bus_indices == i, idx_gen.PG]) / ppc["baseMVA"]
        reactive_power_gen = sum(gen_selector * ppc["gen"][gen_bus_indices == i, idx_gen.QG]) / ppc["baseMVA"]
        voltage_magnitude = ppc["bus"][i, idx_bus.VM]
        voltage_angle_rad = np.deg2rad(ppc["bus"][i, idx_bus.VA])

        if bus_types[i] == BusType.REF:
            m.fix(Vm[i], val=voltage_magnitude)
            m.fix(theta[i], val=voltage_angle_rad)

        if bus_types[i] == BusType.PQ:
            if i in gen_bus_indices:
                m.fix(Pg[i], val=active_power_gen)
                m.fix(Qg[i], val=reactive_power_gen)
            else:
                m.fix(Pg[i], val=0)
                m.fix(Qg[i], val=0)

        if bus_types[i] == BusType.PV:
            m.fix(Vm[i], val=voltage_magnitude)
            if i in gen_bus_indices:
                m.fix(Pg[i], val=active_power_gen)
            else:
                m.fix(Pg[i], val=0)

    # add parameters
    Pd = ppc["bus"][:, idx_bus.PD] / ppc["baseMVA"]
    Qd = ppc["bus"][:, idx_bus.QD] / ppc["baseMVA"]

    load_actions_currently = obs.load_bus - 1
    a_load = load_actions_currently
    load_bus_indices = obs.load_to_subid + load_actions_currently * n_sub
    n_load = load_actions_currently.shape[0]
    for i in range(n_load):
        bus_idx = load_bus_indices[i]
        if i < n_sub:
            Pd[bus_idx] = Pd[bus_idx] * (1 - a_load[i])
        else:
            Pd[bus_idx] = Pd[bus_idx] * a_load[i]

    P = Pg - Pd
    Q = Qg - Qd

    Gbus = Ybus.real.toarray()
    Bbus = Ybus.imag.toarray()

    line_or_actions_currently = obs.line_or_bus - 1

    line_ex_actions_currently = obs.line_ex_bus - 1

    a_line_or = line_or_actions_currently
    a_line_ex = line_ex_actions_currently

    for i in range(nb):
        real_sum_over_k = []
        imag_sum_over_k = []
        for k in range(nb):
            sub_i = i % n_sub
            sub_k = k % n_sub

            # TODO mozda i ide iz line_or_bus, a k iz line_ex_bus

            a_line_i = a_line_or[sub_i]
            a_line_k = a_line_or[sub_k]

            current_a_line_i = line_or_actions_currently[sub_i]
            current_a_line_k = line_or_actions_currently[sub_k]

            if i < n_sub and k < n_sub:
                line_selector = (1 - a_line_i) * (1 - a_line_k)

            elif i < n_sub <= k:
                line_selector = (1 - a_line_i) * a_line_k

            elif k < n_sub <= i:
                line_selector = a_line_i * (1 - a_line_k)

            elif i >= n_sub and k >= n_sub:
                line_selector = a_line_i * a_line_k

            else:
                raise RuntimeError("Something is wrong with the action variable i/k if-else.")

            if current_a_line_i == 0 and current_a_line_k == 0:
                G = Gbus[sub_i, sub_k]
                B = Bbus[sub_i, sub_k]

            elif current_a_line_i == 0 and current_a_line_k == 1:
                G = Gbus[sub_i, sub_k + n_sub]
                B = Bbus[sub_i, sub_k + n_sub]

            elif current_a_line_i == 1 and current_a_line_k == 0:
                G = Gbus[sub_i + n_sub, sub_k]
                B = Bbus[sub_i + n_sub, sub_k]

            elif current_a_line_i == 1 and current_a_line_k == 1:
                G = Gbus[sub_i + n_sub, sub_k + n_sub]
                B = Bbus[sub_i + n_sub, sub_k + n_sub]

            else:
                raise RuntimeError("Something is wrong with the current action i/k if-else.")

            real_sum_over_k.append(
                line_selector * Vm[i] * Vm[k] * (G * m.cos(theta[i] - theta[k]) + B * m.sin(theta[i] - theta[k]))
            )

            imag_sum_over_k.append(
                line_selector * Vm[i] * Vm[k] * (G * m.sin(theta[i] - theta[k]) - B * m.cos(theta[i] - theta[k]))
            )

        m.Equation(0 == -P[i] + sum(real_sum_over_k))
        m.Equation(0 == -Q[i] + sum(imag_sum_over_k))

    # TODO a_gen, a_line, a_load varijable da se naprave
    #  nek se fiksiraju na _currently visto radi testiranja

    # TODO probaj prvo sa ppc-ovima dobijenim od random akcija

    # TODO nejednacine za struje

    variables = PowerFlowVariables(theta=theta, Vm=Vm, Pg=Pg, Qg=Qg)

    m.options.SOLVER = 1
    m.options.RTOL = 1e-8

    m.solve(disp=True)

    # extract values
    theta_gekko, Vm_gekko, Pg_gekko, Qg_gekko = extract_gekko_variables(variables, ppc)

    # compare with pypower solution
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    solved_ppc, success = runpf(ppc, ppopt)

    if success == 0:
        raise ValueError("PYPOWER power flow didn't converge successfully.")

    not_nan_bus = ~np.isnan(solved_ppc["bus"][:, idx_bus.VA])
    not_nan_gen = ~np.isnan(solved_ppc["gen"][:, idx_gen.PG])

    # TODO moram da uzmem u obzir i line_ex_bus

    # fmt: off
    assert all(np.isclose(theta_gekko[not_nan_bus], solved_ppc["bus"][:, idx_bus.VA][not_nan_bus])), "Voltage angles in GEKKO don't match PYPOWER"
    assert all(np.isclose(Vm_gekko[not_nan_bus], solved_ppc["bus"][:, idx_bus.VM][not_nan_bus])), "Voltage magnitudes in GEKKO don't match PYPOWER"
    assert all(np.isclose(Pg_gekko[not_nan_gen], solved_ppc["gen"][:, idx_gen.PG][not_nan_gen])), "Generator active powers in GEKKO don't match PYPOWER"
    assert all(np.isclose(Qg_gekko[not_nan_gen], solved_ppc["gen"][:, idx_gen.QG][not_nan_gen])), "Generator reactive powers in GEKKO don't match PYPOWER"

    If_pu_calc_real = np.zeros((nl,), dtype=np.float64)
    If_pu_calc_imag = np.zeros((nl,), dtype=np.float64)

    Gf = Yf.real.toarray()
    Bf = Yf.imag.toarray()

    theta_gekko_rad = np.deg2rad(theta_gekko)

    for i in range(nl):
        real_sum_over_k = []
        imag_sum_over_k = []
        for k in range(nb):
            sub_k = k % n_sub
            bus_selector = (1 - a_line_or[sub_k]) if k < n_sub else a_line_or[sub_k]

            real_sum_over_k.append(
                bus_selector * Vm_gekko[k] * (Gf[i, k] * np.cos(theta_gekko_rad[k]) - Bf[i, k] * np.sin(theta_gekko_rad[k]))
            )

            imag_sum_over_k.append(
                bus_selector * Vm_gekko[k] * (Gf[i, k] * np.sin(theta_gekko_rad[k]) + Bf[i, k] * np.cos(theta_gekko_rad[k]))
            )

        If_pu_calc_real[i] = sum(real_sum_over_k)
        If_pu_calc_imag[i] = sum(imag_sum_over_k)

    If_pu_abs_calc = np.sqrt(If_pu_calc_real ** 2 + If_pu_calc_imag ** 2)

    If_phase_A_calc = If_pu_abs_calc * base_If_A / np.sqrt(3)

    assert all(np.isclose(If_phase_A_calc, If_phase_A_pypower))


def main():
    # TODO implementirati akcije za generatore in za opterecenja
    #  prvo samo kao float vrednosti
    #  onda sve kao m.Var

    np.random.seed(0)
    random.seed(0)
    env_name = "l2rpn_case14_sandbox"  # for example, other environments might be usable
    env = grid2op.make(env_name)
    env.seed(0)

    random_agent = RandomAgent(env.action_space)
    random_agent.seed(0)

    nb_episode = 1
    for _ in range(nb_episode):
        obs = env.reset()

        reward = env.reward_range[0]
        done = False
        while not done:
            random_action = random_agent.act(obs, reward, done)
            obs, reward, done, info = env.step(random_action)

            ppc = copy.deepcopy(env.backend._grid._ppc)
            verify_equations(ppc, obs)


if __name__ == "__main__":
    main()
