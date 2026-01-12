import grid2op
import numpy as np

# TODO testiraj na brute force agentu
#  i na random agentu
#  tako se menja topologija


def get_bus_busbar_number(bus_id: int, n_sub: int) -> int:
    return bus_id // n_sub + 1


def get_bus_subid(bus_id: int, n_sub: int) -> int:
    return bus_id % n_sub


def get_buses_at_sub(sub_id: int, n_sub: int) -> list[int]:
    return [sub_id, sub_id + n_sub]


def get_bus_voltage(net, bus_id: int) -> tuple[float, float]:
    Vm = net.res_bus["vm_pu"].iloc[bus_id]
    theta = net.res_bus["va_degree"].iloc[bus_id] * np.pi / 180

    if np.isnan(Vm):
        Vm = 0

    if np.isnan(theta):
        theta = 0

    return Vm, theta


def main():
    threshold = 1e-4
    # create an environment
    env_name = "l2rpn_case14_sandbox"  # for example, other environments might be usable
    # env_name = "l2rpn_icaps_2021_small"
    env = grid2op.make(env_name)
    obs = env.reset()
    net = env.backend._grid

    n_sub = env.n_sub
    n_line = env.n_line  # num actual lines + num trafos

    baseMVA = net.sn_mva
    net_line_from_to = tuple(zip(net.line["from_bus"].to_numpy(), net.line["to_bus"].to_numpy()))
    net_trafo_from_to = tuple(zip(net.trafo["hv_bus"].to_numpy(), net.trafo["lv_bus"].to_numpy()))

    # calc admittances
    Yff = np.zeros((n_line,), dtype=np.complex128)
    Yft = np.zeros((n_line,), dtype=np.complex128)
    Ytf = np.zeros((n_line,), dtype=np.complex128)
    Ytt = np.zeros((n_line,), dtype=np.complex128)

    for i in range(n_line):
        sub_from = env.line_or_to_subid[i]
        sub_to = env.line_ex_to_subid[i]

        base_kV = net.bus["vn_kv"].to_numpy()

        # TODO za trafo Vn je onaj na LV strani. Kako znam koje je to?
        #  da l se uopste to cita iz Bus tabele? Mozda je onaj iz trafo tabele

        sub_from_to = (sub_from, sub_to)
        if sub_from_to in net_line_from_to:
            # https://pandapower.readthedocs.io/en/latest/elements/line.html

            z_base = base_kV[sub_to] ** 2 / baseMVA  # pu, the k^2 / M cancels out
            line_idx = net_line_from_to.index(sub_from_to)
            z = (
                (net.line["r_ohm_per_km"] + 1j * net.line["x_ohm_per_km"])
                * net.line["length_km"]
                / net.line["parallel"]
                / z_base
        )[line_idx]
        y = (
                (net.line["g_us_per_km"] * 1e-6 + 1j * 2 * np.pi * net.f_hz * net.line["c_nf_per_km"] * 1e-9)
                * net.line["length_km"]
                * net.line["parallel"]
                * z_base
            )[line_idx]

            Yff[i] = y / 2 + 1 / z
            Yft[i] = -1 / z
            Ytf[i] = -1 / z
            Ytt[i] = y / 2 + 1 / z

        elif sub_from_to in net_trafo_from_to:
            # https://pandapower.readthedocs.io/en/latest/elements/trafo.html

            trafo_idx = net_trafo_from_to.index(sub_from_to)

            z_k = net.trafo["vk_percent"] / 100 * baseMVA / net.trafo["sn_mva"]
            r_k = net.trafo["vkr_percent"] / 100 * baseMVA / net.trafo["sn_mva"]
            x_k = np.sqrt(z_k**2 - r_k**2)
            z = (r_k + 1j * x_k)[trafo_idx]

            y_m_mod = net.trafo["i0_percent"] / 100
            g_m = net.trafo["pfe_kw"] / net.trafo["sn_mva"] / 1000 * baseMVA / net.trafo["sn_mva"]
            b_m = np.sqrt(y_m_mod**2 - g_m**2)
            y = (g_m - 1j * b_m)[trafo_idx]

            assert net.trafo["tap_changer_type"][trafo_idx] in ["Ratio", None]

            tap_side = net.trafo["tap_side"][trafo_idx]
            tap_changer_ratio = (
                1 + (net.trafo["tap_pos"] - net.trafo["tap_neutral"]) * net.trafo["tap_step_percent"] / 100
            )
            trafo_vn_lv_kv = net.trafo["vn_lv_kv"] * tap_changer_ratio if tap_side == "lv" else net.trafo["vn_lv_kv"]
            trafo_vn_hv_kv = net.trafo["vn_hv_kv"] * tap_changer_ratio if tap_side == "hv" else net.trafo["vn_hv_kv"]

            n_mod = trafo_vn_hv_kv / trafo_vn_lv_kv * base_kV[sub_to] / base_kV[sub_from]
            phase_shift = net.trafo["shift_degree"] * np.pi / 180  # rad
            n = (n_mod * np.exp(1j * phase_shift))[trafo_idx]

            Yff[i] = (0.5 * y * z + 1) / (n**2 * z * (0.25 * y * z + 1))
            Yft[i] = -1 / (n * z * (0.25 * y * z + 1))
            Ytf[i] = -1 / (n * z * (0.25 * y * z + 1))
            Ytt[i] = (0.5 * y * z + 1) / (z * (0.25 * y * z + 1))
        else:
            raise ValueError(f"from-to-tuple {sub_from_to} doesn't exist in net.")

    # pretpostavlja se da je prvih n_sub bus 1, a drugih n_bus bus 2

    for bus_id in range(2 * n_sub):
        sub_id = get_bus_subid(bus_id, n_sub=n_sub)  # sub_id [0, n_sub - 1]
        busbar = get_bus_busbar_number(bus_id, n_sub=n_sub)

        # gen
        gen_ids_at_sub = np.argwhere(sub_id == env.gen_to_subid).flatten()
        a_gen = obs.gen_bus[gen_ids_at_sub] - 1  # a_gen [0, 1]
        P_gen = obs.gen_p[gen_ids_at_sub] / baseMVA  # pu
        Q_gen = obs.gen_q[gen_ids_at_sub] / baseMVA

        # load
        load_ids_at_sub = np.argwhere(sub_id == env.load_to_subid).flatten()
        a_load = obs.load_bus[load_ids_at_sub] - 1  # a_load [0, 1]
        P_load = obs.load_p[load_ids_at_sub] / baseMVA  # pu
        Q_load = obs.load_q[load_ids_at_sub] / baseMVA

        shunt_ids_at_sub = np.argwhere(sub_id == env.shunt_to_subid).flatten()
        # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
        g_sh = net.shunt["p_mw"] * net.shunt["step"] / baseMVA
        b_sh = net.shunt["q_mvar"] * net.shunt["step"] / baseMVA
        Vm, _ = get_bus_voltage(net, bus_id)
        Psh = np.sum(Vm**2 * g_sh[shunt_ids_at_sub])
        Qsh = np.sum(Vm**2 * (-b_sh[shunt_ids_at_sub]))

        if busbar == 1:
            Pg_bus = np.sum((1 - a_gen) * P_gen)
            Qg_bus = np.sum((1 - a_gen) * Q_gen)

            Pl_bus = np.sum((1 - a_load) * P_load)
            Ql_bus = np.sum((1 - a_load) * Q_load)

        else:
            Pg_bus = np.sum(a_gen * P_gen)
            Qg_bus = np.sum(a_gen * Q_gen)

            Pl_bus = np.sum(a_load * P_load)
            Ql_bus = np.sum(a_load * Q_load)

        # line or
        line_f_ids_at_sub = np.argwhere(sub_id == obs.line_or_to_subid).flatten()

        Pf = 0.0
        Qf = 0.0
        for line_f_idx in line_f_ids_at_sub:
            to_sub_id = obs.line_ex_to_subid[line_f_idx]
            to_buses = get_buses_at_sub(to_sub_id, n_sub)

            Vm_f, theta_f = get_bus_voltage(net, bus_id)
            Vm_t1, theta_t1 = get_bus_voltage(net, to_buses[0])
            Vm_t2, theta_t2 = get_bus_voltage(net, to_buses[1])

            Yff_r = Yff[line_f_idx].real
            Yff_i = Yff[line_f_idx].imag
            Yft_r = Yft[line_f_idx].real
            Yft_i = Yft[line_f_idx].imag

            a_f = obs.line_or_bus[line_f_idx] - 1
            a_t = obs.line_ex_bus[line_f_idx] - 1

            # fmt: off
            if busbar == 1:
                Pf_line = Vm_f*(1 - a_f)*(Vm_f*(1 - a_f)*(-Yff_i*np.sin(theta_f) + Yff_r*np.cos(theta_f))*np.cos(theta_f) - Vm_f*(1 - a_f)*(-Yff_i*np.cos(theta_f) - Yff_r*np.sin(theta_f))*np.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*np.cos(theta_t1) - Vm_t2*Yft_i*a_t*np.sin(theta_t2) + Vm_t2*Yft_r*a_t*np.cos(theta_t2))*np.cos(theta_f) - (-Vm_t1*Yft_i*(1 - a_t)*np.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*np.sin(theta_t1) - Vm_t2*Yft_i*a_t*np.cos(theta_t2) - Vm_t2*Yft_r*a_t*np.sin(theta_t2))*np.sin(theta_f))  # noqa: E226
                Qf_line = Vm_f*(1 - a_f)*(Vm_f*(1 - a_f)*(-Yff_i*np.sin(theta_f) + Yff_r*np.cos(theta_f))*np.sin(theta_f) + Vm_f*(1 - a_f)*(-Yff_i*np.cos(theta_f) - Yff_r*np.sin(theta_f))*np.cos(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*np.cos(theta_t1) - Vm_t2*Yft_i*a_t*np.sin(theta_t2) + Vm_t2*Yft_r*a_t*np.cos(theta_t2))*np.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*np.sin(theta_t1) - Vm_t2*Yft_i*a_t*np.cos(theta_t2) - Vm_t2*Yft_r*a_t*np.sin(theta_t2))*np.cos(theta_f))  # noqa: E226
            else:
                Pf_line = Vm_f*a_f*(Vm_f*a_f*(-Yff_i*np.sin(theta_f) + Yff_r*np.cos(theta_f))*np.cos(theta_f) - Vm_f*a_f*(-Yff_i*np.cos(theta_f) - Yff_r*np.sin(theta_f))*np.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*np.cos(theta_t1) - Vm_t2*Yft_i*a_t*np.sin(theta_t2) + Vm_t2*Yft_r*a_t*np.cos(theta_t2))*np.cos(theta_f) - (-Vm_t1*Yft_i*(1 - a_t)*np.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*np.sin(theta_t1) - Vm_t2*Yft_i*a_t*np.cos(theta_t2) - Vm_t2*Yft_r*a_t*np.sin(theta_t2))*np.sin(theta_f))  # noqa: E226
                Qf_line = Vm_f*a_f*(Vm_f*a_f*(-Yff_i*np.sin(theta_f) + Yff_r*np.cos(theta_f))*np.sin(theta_f) + Vm_f*a_f*(-Yff_i*np.cos(theta_f) - Yff_r*np.sin(theta_f))*np.cos(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*np.cos(theta_t1) - Vm_t2*Yft_i*a_t*np.sin(theta_t2) + Vm_t2*Yft_r*a_t*np.cos(theta_t2))*np.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*np.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*np.sin(theta_t1) - Vm_t2*Yft_i*a_t*np.cos(theta_t2) - Vm_t2*Yft_r*a_t*np.sin(theta_t2))*np.cos(theta_f))  # noqa: E226
            # fmt: on

            if net.bus["in_service"][bus_id]:
                pf_correct = np.abs(Pf_line - obs.p_or[line_f_idx] / baseMVA) < threshold
                qf_correct = np.abs(Qf_line - obs.q_or[line_f_idx] / baseMVA) < threshold
            else:
                pf_correct = Pf_line == 0
                qf_correct = Qf_line == 0
            assert pf_correct
            assert qf_correct

            Pf += Pf_line
            Qf += Qf_line

        # line ex
        line_t_ids_at_sub = np.argwhere(sub_id == obs.line_ex_to_subid).flatten()

        Pt = 0.0
        Qt = 0.0
        for line_t_idx in line_t_ids_at_sub:
            from_sub_id = obs.line_or_to_subid[line_t_idx]
            from_buses = get_buses_at_sub(from_sub_id, n_sub)

            Vm_t, theta_t = get_bus_voltage(net, bus_id)
            Vm_f1, theta_f1 = get_bus_voltage(net, from_buses[0])
            Vm_f2, theta_f2 = get_bus_voltage(net, from_buses[1])

            Ytf_r = Ytf[line_t_idx].real
            Ytf_i = Ytf[line_t_idx].imag
            Ytt_r = Ytt[line_t_idx].real
            Ytt_i = Ytt[line_t_idx].imag

            a_t = obs.line_ex_bus[line_t_idx] - 1
            a_f = obs.line_or_bus[line_t_idx] - 1

            # fmt: off
            if busbar == 1:
                Pt_line = Vm_t*(1 - a_t)*(Vm_t*(1 - a_t)*(-Ytt_i*np.sin(theta_t) + Ytt_r*np.cos(theta_t))*np.cos(theta_t) - Vm_t*(1 - a_t)*(-Ytt_i*np.cos(theta_t) - Ytt_r*np.sin(theta_t))*np.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*np.cos(theta_f1) - Vm_f2*Ytf_i*a_f*np.sin(theta_f2) + Vm_f2*Ytf_r*a_f*np.cos(theta_f2))*np.cos(theta_t) - (-Vm_f1*Ytf_i*(1 - a_f)*np.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*np.sin(theta_f1) - Vm_f2*Ytf_i*a_f*np.cos(theta_f2) - Vm_f2*Ytf_r*a_f*np.sin(theta_f2))*np.sin(theta_t))  # noqa: E226
                Qt_line = Vm_t*(1 - a_t)*(Vm_t*(1 - a_t)*(-Ytt_i*np.sin(theta_t) + Ytt_r*np.cos(theta_t))*np.sin(theta_t) + Vm_t*(1 - a_t)*(-Ytt_i*np.cos(theta_t) - Ytt_r*np.sin(theta_t))*np.cos(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*np.cos(theta_f1) - Vm_f2*Ytf_i*a_f*np.sin(theta_f2) + Vm_f2*Ytf_r*a_f*np.cos(theta_f2))*np.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*np.sin(theta_f1) - Vm_f2*Ytf_i*a_f*np.cos(theta_f2) - Vm_f2*Ytf_r*a_f*np.sin(theta_f2))*np.cos(theta_t))  # noqa: E226
            else:
                Pt_line = Vm_t*a_t*(Vm_t*a_t*(-Ytt_i*np.sin(theta_t) + Ytt_r*np.cos(theta_t))*np.cos(theta_t) - Vm_t*a_t*(-Ytt_i*np.cos(theta_t) - Ytt_r*np.sin(theta_t))*np.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*np.cos(theta_f1) - Vm_f2*Ytf_i*a_f*np.sin(theta_f2) + Vm_f2*Ytf_r*a_f*np.cos(theta_f2))*np.cos(theta_t) - (-Vm_f1*Ytf_i*(1 - a_f)*np.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*np.sin(theta_f1) - Vm_f2*Ytf_i*a_f*np.cos(theta_f2) - Vm_f2*Ytf_r*a_f*np.sin(theta_f2))*np.sin(theta_t))  # noqa: E226
                Qt_line = Vm_t*a_t*(Vm_t*a_t*(-Ytt_i*np.sin(theta_t) + Ytt_r*np.cos(theta_t))*np.sin(theta_t) + Vm_t*a_t*(-Ytt_i*np.cos(theta_t) - Ytt_r*np.sin(theta_t))*np.cos(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*np.cos(theta_f1) - Vm_f2*Ytf_i*a_f*np.sin(theta_f2) + Vm_f2*Ytf_r*a_f*np.cos(theta_f2))*np.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*np.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*np.sin(theta_f1) - Vm_f2*Ytf_i*a_f*np.cos(theta_f2) - Vm_f2*Ytf_r*a_f*np.sin(theta_f2))*np.cos(theta_t))  # noqa: E226
            # fmt: on

            if net.bus["in_service"][bus_id]:
                pt_correct = np.abs(Pt_line - obs.p_ex[line_t_idx] / baseMVA) < threshold
                qt_correct = np.abs(Qt_line - obs.q_ex[line_t_idx] / baseMVA) < threshold
            else:
                pt_correct = Pt_line == 0
                qt_correct = Qt_line == 0
            assert pt_correct
            assert qt_correct

            Pt += Pt_line
            Qt += Qt_line

        P_balance = Pg_bus - Pl_bus - Psh - Pf - Pt
        Q_balance = Qg_bus - Ql_bus - Qsh - Qf - Qt


        p_balance_correct = np.abs(P_balance) < threshold
        q_balance_correct = np.abs(Q_balance) < threshold

        print(f"{bus_id=}")
        print(f"{P_balance=}")
        print(f"{Q_balance=}")

        assert p_balance_correct
        assert q_balance_correct


if __name__ == "__main__":
    main()
