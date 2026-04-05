import grid2op
import numpy as np
from gekko import GEKKO

from power_flow.validate_equations import calc_admittances, get_bus_subid, get_bus_busbar_number, get_buses_at_sub


def main() -> None:
    # TODO
    #  init grid2op, GEKKO

    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)
    obs = env.reset()

    net = env.backend._grid

    n_sub = env.n_sub
    n_bus = 2 * n_sub

    n_gen = env.n_gen
    n_load = env.n_load
    n_line = env.n_line

    m = GEKKO(remote=False)

    # define variables

    Vm = m.Array(m.Var, n_bus, lb=0, value=1)
    theta = m.Array(m.Var, n_bus, lb=-np.pi, ub=np.pi)

    Pg = m.Array(m.Var, n_gen)
    Qg = m.Array(m.Var, n_gen)

    a_gen = m.Array(m.Var, n_gen, integer=True, lb=0, ub=1)
    a_load = m.Array(m.Var, n_load, integer=True, lb=0, ub=1)
    a_or = m.Array(m.Var, n_line, integer=True, lb=0, ub=1)
    a_ex = m.Array(m.Var, n_line, integer=True, lb=0, ub=1)

    n_sub = env.n_sub
    n_bus = 2 * n_sub
    baseMVA = net.sn_mva
    Yff, Yft, Ytf, Ytt = calc_admittances(env, net)

    residuals = []
    for bus_id in range(n_bus):

        if np.isnan(net.res_bus.vm_pu[bus_id]):
            # TODO not the right was to handle
            #  cause a bus might switch on
            continue

        sub_id = get_bus_subid(bus_id, n_sub=n_sub)
        busbar = get_bus_busbar_number(bus_id, n_sub=n_sub)

        # ── Generators ────────────────────────────────────────────────────────
        gen_ids = np.argwhere(sub_id == env.gen_to_subid).flatten()
        if busbar == 1:
            Pg_bus = m.sum([Pg[g] * (1 - a_gen[g]) for g in gen_ids])
            Qg_bus = m.sum([Qg[g] * (1 - a_gen[g]) for g in gen_ids])
        else:
            Pg_bus = m.sum([Pg[g] * a_gen[g] for g in gen_ids])
            Qg_bus = m.sum([Qg[g] * a_gen[g] for g in gen_ids])

        # ── Loads (constants, not variables) ──────────────────────────────────
        P_load = obs.load_p / baseMVA  # shape: (n_load,)
        Q_load = obs.load_q / baseMVA
        load_ids = np.argwhere(sub_id == env.load_to_subid).flatten()

        if busbar == 1:
            Pl_bus = m.sum([P_load[load_id] * (1 - a_load[load_id]) for load_id in load_ids])
            Ql_bus = m.sum([Q_load[load_id] * (1 - a_load[load_id]) for load_id in load_ids])
        else:
            Pl_bus = m.sum([P_load[load_id] * a_load[load_id] for load_id in load_ids])
            Ql_bus = m.sum([Q_load[load_id] * a_load[load_id] for load_id in load_ids])

        # ── Shunts (fully constant) ────────────────────────────────────────────
        shunt_ids = np.argwhere(net.shunt["bus"] == bus_id).flatten()
        g_sh = (net.shunt["p_mw"] * net.shunt["step"] / baseMVA * net.shunt["in_service"]).values
        b_sh = (net.shunt["q_mvar"] * net.shunt["step"] / baseMVA * net.shunt["in_service"]).values
        Psh = m.Intermediate(m.sum([Vm[bus_id] ** 2 * g_sh[s] for s in shunt_ids]))
        Qsh = m.Intermediate(m.sum([Vm[bus_id] ** 2 * (-b_sh[s]) for s in shunt_ids]))

        # ── Helper intermediates for bus_id ───────────────────────────────────
        Vm_f = Vm[bus_id]
        th_f = theta[bus_id]
        cos_f = m.Intermediate(m.cos(th_f))
        sin_f = m.Intermediate(m.sin(th_f))

        # ── Lines (from side) ─────────────────────────────────────────────────
        Pf_total = 0
        Qf_total = 0
        line_f_ids = np.argwhere(sub_id == obs.line_or_to_subid).flatten()

        for line_idx in line_f_ids:
            to_sub_id = obs.line_ex_to_subid[line_idx]
            to_buses = get_buses_at_sub(to_sub_id, n_sub)  # [bus_t1, bus_t2]

            Yff_r = Yff[line_idx].real;
            Yff_i = Yff[line_idx].imag
            Yft_r = Yft[line_idx].real;
            Yft_i = Yft[line_idx].imag

            a_f = a_or[line_idx]
            a_t = a_ex[line_idx]
            af = m.Intermediate(a_f if busbar == 2 else (1 - a_f))  # weight for this busbar

            # Voltages at the two possible "to" buses
            Vm_t1, th_t1 = Vm[to_buses[0]], theta[to_buses[0]]
            Vm_t2, th_t2 = Vm[to_buses[1]], theta[to_buses[1]]

            cos_t1 = m.Intermediate(m.cos(th_t1));
            sin_t1 = m.Intermediate(m.sin(th_t1))
            cos_t2 = m.Intermediate(m.cos(th_t2));
            sin_t2 = m.Intermediate(m.sin(th_t2))

            # "From" self term:  Yff * Vf^2  (real and imag parts of current)
            Iself_r = m.Intermediate(Yff_r * cos_f - Yff_i * sin_f)  # Re(Yff * e^{j*th_f})
            Iself_i = m.Intermediate(Yff_r * sin_f + Yff_i * cos_f)  # Im(Yff * e^{j*th_f})

            # "To" mutual term:  Yft * Vt  (weighted sum over both busbars)
            Imut_r = m.Intermediate(
                Yft_r * (Vm_t1 * (1 - a_t) * cos_t1 + Vm_t2 * a_t * cos_t2)
                - Yft_i * (Vm_t1 * (1 - a_t) * sin_t1 + Vm_t2 * a_t * sin_t2)
            )
            Imut_i = m.Intermediate(
                Yft_r * (Vm_t1 * (1 - a_t) * sin_t1 + Vm_t2 * a_t * sin_t2)
                + Yft_i * (Vm_t1 * (1 - a_t) * cos_t1 + Vm_t2 * a_t * cos_t2)
            )

            # S = Vf * conj(I);  P = Re(S), Q = Im(S)
            #   I_total = (Iself_r + Imut_r) + j*(Iself_i + Imut_i)
            #   Vf = Vm_f*(cos_f + j*sin_f)
            #   P  = Vm_f * [cos_f*(Iself_r+Imut_r) + sin_f*(Iself_i+Imut_i)]   <- conj flips Im
            #   Q  = Vm_f * [sin_f*(Iself_r+Imut_r) - cos_f*(Iself_i+Imut_i)]

            Vf_weighted = m.Intermediate(Vm_f * af)  # zero-out if wrong busbar

            Pf_line = m.Intermediate(
                Vf_weighted * (
                        cos_f * (Vm_f * af * Iself_r + Imut_r)
                        + sin_f * (Vm_f * af * Iself_i + Imut_i)
                )
            )
            Qf_line = m.Intermediate(
                Vf_weighted * (
                        sin_f * (Vm_f * af * Iself_r + Imut_r)
                        - cos_f * (Vm_f * af * Iself_i + Imut_i)
                )
            )

            Pf_total = Pf_total + Pf_line
            Qf_total = Qf_total + Qf_line

        # ── Lines (to side) ───────────────────────────────────────────────────
        Pt_total = 0
        Qt_total = 0
        line_t_ids = np.argwhere(sub_id == obs.line_ex_to_subid).flatten()

        for line_idx in line_t_ids:
            from_sub_id = obs.line_or_to_subid[line_idx]
            from_buses = get_buses_at_sub(from_sub_id, n_sub)

            Ytf_r = Ytf[line_idx].real
            Ytf_i = Ytf[line_idx].imag
            Ytt_r = Ytt[line_idx].real
            Ytt_i = Ytt[line_idx].imag

            a_t = a_ex[line_idx]
            a_f = a_or[line_idx]
            at = m.Intermediate(a_t if busbar == 2 else (1 - a_t))

            Vm_f1, th_f1 = Vm[from_buses[0]], theta[from_buses[0]]
            Vm_f2, th_f2 = Vm[from_buses[1]], theta[from_buses[1]]

            cos_f1 = m.Intermediate(m.cos(th_f1))
            sin_f1 = m.Intermediate(m.sin(th_f1))
            cos_f2 = m.Intermediate(m.cos(th_f2))
            sin_f2 = m.Intermediate(m.sin(th_f2))

            # "To" self term
            Iself_r = m.Intermediate(Ytt_r * cos_f - Ytt_i * sin_f)
            Iself_i = m.Intermediate(Ytt_r * sin_f + Ytt_i * cos_f)

            # "From" mutual term
            Imut_r = m.Intermediate(
                Ytf_r * (Vm_f1 * (1 - a_f) * cos_f1 + Vm_f2 * a_f * cos_f2)
                - Ytf_i * (Vm_f1 * (1 - a_f) * sin_f1 + Vm_f2 * a_f * sin_f2)
            )
            Imut_i = m.Intermediate(
                Ytf_r * (Vm_f1 * (1 - a_f) * sin_f1 + Vm_f2 * a_f * sin_f2)
                + Ytf_i * (Vm_f1 * (1 - a_f) * cos_f1 + Vm_f2 * a_f * cos_f2)
            )

            Vt_weighted = m.Intermediate(Vm_f * at)

            Pt_line = m.Intermediate(
                Vt_weighted * (
                        cos_f * (Vm_f * at * Iself_r + Imut_r)
                        + sin_f * (Vm_f * at * Iself_i + Imut_i)
                )
            )
            Qt_line = m.Intermediate(
                Vt_weighted * (
                        sin_f * (Vm_f * at * Iself_r + Imut_r)
                        - cos_f * (Vm_f * at * Iself_i + Imut_i)
                )
            )

            Pt_total = Pt_total + Pt_line
            Qt_total = Qt_total + Qt_line

        # ── Power balance constraints ──────────────────────────────────────────
        P_res = m.Intermediate(Pg_bus - Pl_bus + Psh - Pf_total - Pt_total)
        Q_res = m.Intermediate(Qg_bus - Ql_bus + Qsh - Qf_total - Qt_total)
        residuals.append((bus_id, P_res, Q_res))
        m.Equation(P_res == 0)
        m.Equation(Q_res == 0)

    # TODO test
    #  fix all the variables to the ground truth values and see if all the equations are satisfied

    # TODO handle disconnected buses

    def fix(var, value):
        var.value = value
        var.LOWER = value
        var.UPPER = value

    # Fix voltage variables
    for i in range(n_bus):
        if np.isnan(net.res_bus.vm_pu[i]):
            # Mirror from the corresponding busbar 1
            busbar1_id = i - n_sub  # since busbar 2 = busbar 1 index + n_sub
            fix(Vm[i], net.res_bus.vm_pu[busbar1_id])
            fix(theta[i], np.deg2rad(net.res_bus.va_degree[busbar1_id]))
        else:
            fix(Vm[i], net.res_bus.vm_pu[i])
            fix(theta[i], np.deg2rad(net.res_bus.va_degree[i]))

    # Fix generator dispatch

    # TODO am I extrcting the actions correctly?
    # TODO am I not screwing up the units?
    for i in range(n_gen):
        fix(Pg[i], net.res_gen.p_mw[i] / baseMVA)
        fix(Qg[i], net.res_gen.q_mvar[i] / baseMVA)

    # Fix binary switching variables
    for i in range(n_gen):
        fix(a_gen[i], obs.gen_bus[i] - 1)

    for i in range(n_load):
        fix(a_load[i], obs.load_bus[i] - 1)

    for i in range(n_line):
        fix(a_or[i], obs.line_or_bus[i] - 1)
        fix(a_ex[i], obs.line_ex_bus[i] - 1)

    # After fixing, print how many are actually fixed
    fixed = sum(1 for i in range(n_bus) if Vm[i].LOWER == Vm[i].UPPER)
    print(f"Fixed Vm: {fixed} / {n_bus}")

    m.options.SOLVER = 1  # APOPT (needed for integer vars)
    m.options.IMODE = 3  # steady-state optimization

    m.Minimize(0)  # no objective, just check constraints
    m.solve(disp=True)

    # TODO
    #  for each bus see if there's an ambiguity in the bus type
    #  fix variables if it's certain which type it is
    #  otherwise add the equality/inequality constraints needed to handle the ambiguity

    # TODO test
    #  fix all the variables to the ground truth values and see if all the equations are satisfied
    print(f"n_bus={n_bus}, n_gen={n_gen}, n_load={n_load}, n_line={n_line}")
    print(
        f"Expected variables: Vm={n_bus} + theta={n_bus} + Pg={n_gen} + Qg={n_gen} + a_gen={n_gen} + a_load={n_load} + a_or={n_line} + a_ex={n_line}")
    print(f"Total expected: {n_bus + n_bus + n_gen + n_gen + n_gen + n_load + n_line + n_line}")
    print(f"GEKKO reports: 245")

    # Print intermediate values to verify balance equations
    print("Checking power balance residuals...")
    for bus_id, P_res, Q_res in residuals:
        print(f"bus {bus_id}: P_res={P_res.value[0]:.6f}, Q_res={Q_res.value[0]:.6f}")

if __name__ == "__main__":
    main()