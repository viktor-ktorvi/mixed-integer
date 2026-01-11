import grid2op
import numpy as np
import pandas as pd


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


def calculate_trafo_parameters(net, idx):
    """
    Calculate Z and Y parameters for a T-type transformer model from pandapower net.
    Uses the exact equations from pandapower documentation.
    Returns values in the network per-unit system.

    Parameters:
    -----------
    net : pandapowerNet
        pandapower network object
    idx : int
        Index of the transformer in net.trafo

    Returns:
    --------
    dict : Dictionary containing Z and Y parameters as used in pandapower
    """

    # Get transformer data from the net object
    trafo = net.trafo.loc[idx]

    vn_hv_kv = trafo['vn_hv_kv']
    vn_lv_kv = trafo['vn_lv_kv']
    sn_mva = trafo['sn_mva']
    vk_percent = trafo['vk_percent']
    vkr_percent = trafo['vkr_percent']
    pfe_kw = trafo['pfe_kw']
    i0_percent = trafo['i0_percent']
    shift_degree = trafo['shift_degree']

    # Get bus information
    hv_bus = trafo['hv_bus']
    lv_bus = trafo['lv_bus']
    vn_hv_bus_kv = net.bus.loc[hv_bus, 'vn_kv']
    vn_lv_bus_kv = net.bus.loc[lv_bus, 'vn_kv']

    # Network rated apparent power (system-wide)
    sn_mva_net = net.sn_mva if hasattr(net, 'sn_mva') else 1.0  # Default 1 MVA if not set

    # Tap changer parameters
    tap_side = trafo.get('tap_side', None)
    tap_pos = trafo.get('tap_pos', trafo.get('tap_neutral', np.nan))
    tap_neutral = trafo.get('tap_neutral', np.nan)
    tap_step_percent = trafo.get('tap_step_percent', np.nan)
    tap_step_degree = trafo.get('tap_step_degree', np.nan)

    # Calculate tap factor
    if pd.notna(tap_pos) and pd.notna(tap_neutral) and pd.notna(tap_step_percent):
        n_tap = 1 + (tap_pos - tap_neutral) * (tap_step_percent / 100)
    else:
        n_tap = 1.0

    # Apply tap factor to appropriate side
    if tap_side == 'hv':
        vn_hv_corrected = vn_hv_kv * n_tap
        vn_lv_corrected = vn_lv_kv
    elif tap_side == 'lv':
        vn_hv_corrected = vn_hv_kv
        vn_lv_corrected = vn_lv_kv * n_tap
    else:
        vn_hv_corrected = vn_hv_kv
        vn_lv_corrected = vn_lv_kv

    # Transformer ratio (magnitude)
    n = vn_hv_corrected / vn_lv_corrected

    # Phase shift angle (total)
    theta_deg = shift_degree
    if pd.notna(tap_step_degree) and pd.notna(tap_pos) and pd.notna(tap_neutral):
        theta_tap = tap_step_degree * (tap_pos - tap_neutral)
        theta_deg += theta_tap

    theta_rad = np.radians(theta_deg)

    # Complex transformer ratio
    n_complex = n * np.exp(1j * theta_rad)

    # ==== Z Parameters in transformer's own per-unit system ====
    # (per unit on transformer rating: vn_lv_kv^2 / sn_mva)
    z_k = vk_percent / 100  # per unit on transformer base
    r_k = vkr_percent / 100  # per unit on transformer base
    x_k = np.sqrt(z_k ** 2 - r_k ** 2)

    # Complex series impedance (transformer per unit)
    z_trafo_pu = r_k + 1j * x_k

    # ==== Y Parameters in transformer's own per-unit system ====
    y_m = i0_percent / 100  # per unit magnitude
    g_m = (pfe_kw / 1000) / sn_mva  # per unit conductance
    b_m = np.sqrt(y_m ** 2 - g_m ** 2)  # per unit susceptance

    # Complex shunt admittance (transformer per unit)
    y_trafo_pu = g_m - 1j * b_m

    # ==== Convert to Network Per-Unit System ====

    # Network base impedance (at LV side, which is reference for transformer)
    # Z_N = V_N^2 / S_N where V_N is the bus nominal voltage
    z_base_network_lv = vn_lv_bus_kv ** 2 / sn_mva_net

    # Transformer reference impedance (at LV side)
    # Z_ref,trafo = vn_lv_kv^2 * net.sn_mva / sn_mva
    z_ref_trafo = (vn_lv_kv ** 2) / sn_mva

    # Conversion factor for impedance
    # z = z_k * (Z_ref,trafo / Z_N)
    z_conversion = z_ref_trafo / z_base_network_lv

    # Conversion factor for admittance
    # y = y_m * (Z_N / Z_ref,trafo)
    y_conversion = z_base_network_lv / z_ref_trafo

    # Convert Z to network per unit
    z_network_pu = z_trafo_pu * z_conversion

    # Convert Y to network per unit
    y_network_pu = y_trafo_pu * y_conversion

    return z_network_pu, y_network_pu


def main():
    num_dec = 4
    # create an environment
    env_name = "l2rpn_case14_sandbox"  # for example, other environments might be usable
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
            tap_changer_ratio = 1 + (net.trafo["tap_pos"] - net.trafo["tap_neutral"]) * net.trafo["tap_step_percent"] / 100
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

    bus_id = 3  # for each bus
    for bus_id in range(n_sub, 2*n_sub):
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
        Psh = np.sum(Vm ** 2 * g_sh[shunt_ids_at_sub])
        Qsh = np.sum(Vm ** 2 * (-b_sh[shunt_ids_at_sub]))

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
        print(f"{line_f_ids_at_sub=}")

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

            pf_correct = round(Pf_line, num_dec) == round(obs.p_or[line_f_idx] / baseMVA, num_dec)
            qf_correct = round(Qf_line, num_dec) == round(obs.q_or[line_f_idx] / baseMVA, num_dec)
            assert pf_correct
            assert qf_correct


            Pf += Pf_line
            Qf += Qf_line

        # line ex
        line_t_ids_at_sub = np.argwhere(sub_id == obs.line_ex_to_subid).flatten()

        print(f"{line_t_ids_at_sub=}")
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

            # TODO nesto sitno ne valja kod modelovanja trafoa
            pt_correct = round(Pt_line, num_dec) == round(obs.p_ex[line_t_idx] / baseMVA, num_dec)
            qt_correct = round(Qt_line, num_dec) == round(obs.q_ex[line_t_idx] / baseMVA, num_dec)
            assert pt_correct
            assert qt_correct

            Pt += Pt_line
            Qt += Qt_line

        P_balance = Pg_bus - Pl_bus - Psh - Pf - Pt
        Q_balance = Qg_bus - Ql_bus - Qsh - Qf - Qt

        p_balance_correct = P_balance < 1e-6
        q_balance_correct = Q_balance < 1e-6

        print(f"{P_balance=}")
        print(f"{Q_balance=}")

        assert p_balance_correct
        assert q_balance_correct


if __name__ == "__main__":
    main()
