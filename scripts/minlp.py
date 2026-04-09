import grid2op
import numpy as np
from gekko import GEKKO
from grid2op import Environment, Observation

from power_flow.validate_equations import (
    calc_admittances,
    get_bus_busbar_number,
    get_bus_subid,
    get_buses_at_sub,
)


def get_gen_ids_at_sub(sub_id: int, env: Environment) -> np.ndarray:
    return np.argwhere(sub_id == env.gen_to_subid).flatten()


def get_grid_sizes(env: Environment) -> tuple[int, int, int, int, int]:
    n_sub = env.n_sub
    n_bus = 2 * n_sub

    n_gen = env.n_gen
    n_load = env.n_load
    n_line = env.n_line
    return n_sub, n_bus, n_gen, n_load, n_line


class MINLP:
    def __init__(self, env: Environment, obs: Observation) -> None:
        self.env = env
        self.obs = obs

        self.m = GEKKO(remote=False)

        self.n_sub, self.n_bus, self.n_gen, self.n_load, self.n_line = get_grid_sizes(env)

        self.Vm = self.m.Array(self.m.Var, self.n_bus, lb=0, value=1)
        self.theta = self.m.Array(self.m.Var, self.n_bus, lb=-np.pi, ub=np.pi)

        self.Pg = self.m.Array(self.m.Var, self.n_gen)
        self.Qg = self.m.Array(self.m.Var, self.n_gen)

        self.a_gen = self.m.Array(self.m.Var, self.n_gen, integer=True, lb=0, ub=1)
        self.a_load = self.m.Array(self.m.Var, self.n_load, integer=True, lb=0, ub=1)
        self.a_or = self.m.Array(self.m.Var, self.n_line, integer=True, lb=0, ub=1)
        self.a_ex = self.m.Array(self.m.Var, self.n_line, integer=True, lb=0, ub=1)

        self.M_vm: float = 0.2
        self.M_th: float = np.pi

        self.net = env.backend._grid

        self.baseMVA = self.net.sn_mva
        self.Yff, self.Yft, self.Ytf, self.Ytt = calc_admittances(self.env, self.net)

        self.residuals = []

    def add_bus_type_constraints(self):
        for bus_id in range(self.n_bus):
            sub_id = get_bus_subid(bus_id, n_sub=self.n_sub)
            busbar = get_bus_busbar_number(bus_id, n_sub=self.n_sub)

            gen_ids = get_gen_ids_at_sub(sub_id, self.env)

            if len(gen_ids) > 0:
                # for fixing the voltage, take the 1st generators voltage
                Vm_res = self.net.res_gen.vm_pu[gen_ids[0]]
                theta_res = np.deg2rad(self.net.res_gen.va_degree[gen_ids[0]])

            if any(self.net.gen.slack[gen_ids]):
                # busbar is either slack or PQ depending on the switching of the slack generator
                if len(gen_ids) != 1:
                    raise RuntimeError("Expected one generator on slack substation.")

                # Pg and Qg remain variables (we don't do anything)

                slack_gen_id = gen_ids[0]

                if busbar == 1:
                    a_gs = self.a_gen[slack_gen_id]
                else:
                    a_gs = 1 - self.a_gen[slack_gen_id]

                # Vm and theta are fixed if the slack gen is connected to the bus (slack), otherwise variables (PQ)

                self.m.Equation(self.Vm[bus_id] >= Vm_res - self.M_vm * a_gs)
                self.m.Equation(self.Vm[bus_id] <= Vm_res + self.M_vm * a_gs)

                self.m.Equation(self.theta[bus_id] >= theta_res - self.M_th * a_gs)
                self.m.Equation(self.theta[bus_id] <= theta_res + self.M_th * a_gs)

                continue

            if len(gen_ids) > 0:
                # bus is either PV or PQ based on if there are any generators connected to it

                # fix all the generator active powers, because they're gonna be attached to PV buses anyhow
                self.m.fix(self.Pg[gen_ids], val=self.net.res_gen.p_mw[gen_ids] / self.baseMVA)

                # Qg remains free
                # theta remains free

                # Vm is fixed if there's at least one generator connected to the busbar

                if busbar == 1:
                    a_g = self.a_gen[gen_ids]
                else:
                    a_g = 1 - self.a_gen[gen_ids]

                n_g = len(a_g)

                z = self.m.Var(integer=True, lb=0, ub=1)  # z == 0 if at least one generator is at busbar
                for a_gi in a_g:
                    self.m.Equation(z <= a_gi)

                self.m.Equation(z >= self.m.sum(a_g) - (n_g - 1))

                self.m.Equation(self.Vm[bus_id] >= Vm_res - self.M_vm * z)
                self.m.Equation(self.Vm[bus_id] <= Vm_res + self.M_vm * z)

    def add_power_flow_equations(self):
        for bus_id in range(self.n_bus):
            # TODO how does conservation of power work on disconnected buses?
            #  does it need handling or does it sort itself out?

            sub_id = get_bus_subid(bus_id, n_sub=self.n_sub)
            busbar = get_bus_busbar_number(bus_id, n_sub=self.n_sub)

            # ── Generators ────────────────────────────────────────────────────────
            gen_ids = get_gen_ids_at_sub(sub_id, self.env)
            if busbar == 1:
                Pg_bus = self.m.sum([self.Pg[g] * (1 - self.a_gen[g]) for g in gen_ids])
                Qg_bus = self.m.sum([self.Qg[g] * (1 - self.a_gen[g]) for g in gen_ids])
            else:
                Pg_bus = self.m.sum([self.Pg[g] * self.a_gen[g] for g in gen_ids])
                Qg_bus = self.m.sum([self.Qg[g] * self.a_gen[g] for g in gen_ids])

            # ── Loads (constants, not variables) ──────────────────────────────────
            P_load = self.obs.load_p / self.baseMVA  # shape: (n_load,)
            Q_load = self.obs.load_q / self.baseMVA
            load_ids = np.argwhere(sub_id == self.env.load_to_subid).flatten()

            # TODO ovo se moze preraditi da se a i 1-a zamene jednom promenljivom
            if busbar == 1:
                Pl_bus = self.m.sum([P_load[load_id] * (1 - self.a_load[load_id]) for load_id in load_ids])
                Ql_bus = self.m.sum([Q_load[load_id] * (1 - self.a_load[load_id]) for load_id in load_ids])
            else:
                Pl_bus = self.m.sum([P_load[load_id] * self.a_load[load_id] for load_id in load_ids])
                Ql_bus = self.m.sum([Q_load[load_id] * self.a_load[load_id] for load_id in load_ids])

            # ── Shunts (fully constant) ────────────────────────────────────────────
            shunt_ids = np.argwhere(self.net.shunt["bus"] == bus_id).flatten()
            g_sh = (
                self.net.shunt["p_mw"] * self.net.shunt["step"] / self.baseMVA * self.net.shunt["in_service"]
            ).values
            b_sh = (
                self.net.shunt["q_mvar"] * self.net.shunt["step"] / self.baseMVA * self.net.shunt["in_service"]
            ).values
            Psh = self.m.Intermediate(self.m.sum([self.Vm[bus_id] ** 2 * g_sh[s] for s in shunt_ids]))
            Qsh = self.m.Intermediate(self.m.sum([self.Vm[bus_id] ** 2 * (-b_sh[s]) for s in shunt_ids]))

            # ── Helper intermediates for bus_id ───────────────────────────────────
            Vm_f = self.Vm[bus_id]
            th_f = self.theta[bus_id]
            cos_f = self.m.Intermediate(self.m.cos(th_f))
            sin_f = self.m.Intermediate(self.m.sin(th_f))

            # ── Lines (from side) ─────────────────────────────────────────────────
            Pf_total = 0
            Qf_total = 0
            line_f_ids = np.argwhere(sub_id == self.obs.line_or_to_subid).flatten()

            for line_idx in line_f_ids:
                to_sub_id = self.obs.line_ex_to_subid[line_idx]
                to_buses = get_buses_at_sub(to_sub_id, self.n_sub)  # [bus_t1, bus_t2]

                Yff_r = self.Yff[line_idx].real
                Yff_i = self.Yff[line_idx].imag
                Yft_r = self.Yft[line_idx].real
                Yft_i = self.Yft[line_idx].imag

                a_f = self.a_or[line_idx]
                a_t = self.a_ex[line_idx]
                af = self.m.Intermediate(a_f if busbar == 2 else (1 - a_f))  # weight for this busbar

                # Voltages at the two possible "to" buses
                Vm_t1, th_t1 = self.Vm[to_buses[0]], self.theta[to_buses[0]]
                Vm_t2, th_t2 = self.Vm[to_buses[1]], self.theta[to_buses[1]]

                cos_t1 = self.m.Intermediate(self.m.cos(th_t1))
                sin_t1 = self.m.Intermediate(self.m.sin(th_t1))
                cos_t2 = self.m.Intermediate(self.m.cos(th_t2))
                sin_t2 = self.m.Intermediate(self.m.sin(th_t2))

                # "From" self term:  Yff * Vf^2  (real and imag parts of current)
                Iself_r = self.m.Intermediate(Yff_r * cos_f - Yff_i * sin_f)  # Re(Yff * e^{j*th_f})
                Iself_i = self.m.Intermediate(Yff_r * sin_f + Yff_i * cos_f)  # Im(Yff * e^{j*th_f})

                # "To" mutual term:  Yft * Vt  (weighted sum over both busbars)
                Imut_r = self.m.Intermediate(
                    Yft_r * (Vm_t1 * (1 - a_t) * cos_t1 + Vm_t2 * a_t * cos_t2)
                    - Yft_i * (Vm_t1 * (1 - a_t) * sin_t1 + Vm_t2 * a_t * sin_t2)
                )
                Imut_i = self.m.Intermediate(
                    Yft_r * (Vm_t1 * (1 - a_t) * sin_t1 + Vm_t2 * a_t * sin_t2)
                    + Yft_i * (Vm_t1 * (1 - a_t) * cos_t1 + Vm_t2 * a_t * cos_t2)
                )

                # S = Vf * conj(I);  P = Re(S), Q = Im(S)
                #   I_total = (Iself_r + Imut_r) + j*(Iself_i + Imut_i)
                #   Vf = Vm_f*(cos_f + j*sin_f)
                #   P  = Vm_f * [cos_f*(Iself_r+Imut_r) + sin_f*(Iself_i+Imut_i)]   <- conj flips Im
                #   Q  = Vm_f * [sin_f*(Iself_r+Imut_r) - cos_f*(Iself_i+Imut_i)]

                Vf_weighted = self.m.Intermediate(Vm_f * af)  # zero-out if wrong busbar

                If_total_r = Vm_f * af * Iself_r + Imut_r
                If_total_i = Vm_f * af * Iself_i + Imut_i
                Pf_line = self.m.Intermediate(Vf_weighted * (cos_f * If_total_r + sin_f * If_total_i))
                Qf_line = self.m.Intermediate(Vf_weighted * (sin_f * If_total_r - cos_f * If_total_i))

                Pf_total = Pf_total + Pf_line
                Qf_total = Qf_total + Qf_line

            # ── Lines (to side) ───────────────────────────────────────────────────
            Pt_total = 0
            Qt_total = 0
            line_t_ids = np.argwhere(sub_id == self.obs.line_ex_to_subid).flatten()

            for line_idx in line_t_ids:
                from_sub_id = self.obs.line_or_to_subid[line_idx]
                from_buses = get_buses_at_sub(from_sub_id, self.n_sub)

                Ytf_r = self.Ytf[line_idx].real
                Ytf_i = self.Ytf[line_idx].imag
                Ytt_r = self.Ytt[line_idx].real
                Ytt_i = self.Ytt[line_idx].imag

                a_t = self.a_ex[line_idx]
                a_f = self.a_or[line_idx]
                at = self.m.Intermediate(a_t if busbar == 2 else (1 - a_t))

                Vm_f1, th_f1 = self.Vm[from_buses[0]], self.theta[from_buses[0]]
                Vm_f2, th_f2 = self.Vm[from_buses[1]], self.theta[from_buses[1]]

                cos_f1 = self.m.Intermediate(self.m.cos(th_f1))
                sin_f1 = self.m.Intermediate(self.m.sin(th_f1))
                cos_f2 = self.m.Intermediate(self.m.cos(th_f2))
                sin_f2 = self.m.Intermediate(self.m.sin(th_f2))

                # "To" self term
                Iself_r = self.m.Intermediate(Ytt_r * cos_f - Ytt_i * sin_f)
                Iself_i = self.m.Intermediate(Ytt_r * sin_f + Ytt_i * cos_f)

                # "From" mutual term
                Imut_r = self.m.Intermediate(
                    Ytf_r * (Vm_f1 * (1 - a_f) * cos_f1 + Vm_f2 * a_f * cos_f2)
                    - Ytf_i * (Vm_f1 * (1 - a_f) * sin_f1 + Vm_f2 * a_f * sin_f2)
                )
                Imut_i = self.m.Intermediate(
                    Ytf_r * (Vm_f1 * (1 - a_f) * sin_f1 + Vm_f2 * a_f * sin_f2)
                    + Ytf_i * (Vm_f1 * (1 - a_f) * cos_f1 + Vm_f2 * a_f * cos_f2)
                )

                Vt_weighted = self.m.Intermediate(Vm_f * at)

                Pt_line = self.m.Intermediate(
                    Vt_weighted * (cos_f * (Vm_f * at * Iself_r + Imut_r) + sin_f * (Vm_f * at * Iself_i + Imut_i))
                )
                Qt_line = self.m.Intermediate(
                    Vt_weighted * (sin_f * (Vm_f * at * Iself_r + Imut_r) - cos_f * (Vm_f * at * Iself_i + Imut_i))
                )

                Pt_total = Pt_total + Pt_line
                Qt_total = Qt_total + Qt_line

            # ── Power balance constraints ──────────────────────────────────────────
            P_res = self.m.Intermediate(Pg_bus - Pl_bus + Psh - Pf_total - Pt_total)
            Q_res = self.m.Intermediate(Qg_bus - Ql_bus + Qsh - Qf_total - Qt_total)
            self.residuals.append((bus_id, P_res, Q_res))
            self.m.Equation(P_res == 0)
            self.m.Equation(Q_res == 0)


def main() -> None:
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)
    obs = env.reset()

    problem = MINLP(env, obs)
    problem.add_bus_type_constraints()
    problem.add_power_flow_equations()

    # TODO where is the current < current limit part?

    # TODO make this formulation it's own function
    #  add in the existing test fixture setup, fix all the variables to the ground truth and make sure energy is conserved

    # TODO extract the action dict from the solutions

    # TODO finally, run an episode while solving the problem
    #  check if the line utilizations are really bellow the threshold

    # def fix(var, value):
    #     var.value = value
    #     var.LOWER = value
    #     var.UPPER = value
    #
    # # Fix voltage variables
    # for i in range(n_bus):
    #     if np.isnan(net.res_bus.vm_pu[i]):
    #         # Mirror from the corresponding busbar 1
    #         busbar1_id = i - n_sub  # since busbar 2 = busbar 1 index + n_sub
    #         fix(Vm[i], net.res_bus.vm_pu[busbar1_id])
    #         fix(theta[i], np.deg2rad(net.res_bus.va_degree[busbar1_id]))
    #     else:
    #         fix(Vm[i], net.res_bus.vm_pu[i])
    #         fix(theta[i], np.deg2rad(net.res_bus.va_degree[i]))
    #
    # for i in range(n_gen):
    #     fix(Pg[i], net.res_gen.p_mw[i] / baseMVA)
    #     fix(Qg[i], net.res_gen.q_mvar[i] / baseMVA)
    #
    # # Fix binary switching variables
    # for i in range(n_gen):
    #     fix(a_gen[i], obs.gen_bus[i] - 1)
    #
    # for i in range(n_load):
    #     fix(a_load[i], obs.load_bus[i] - 1)
    #
    # for i in range(n_line):
    #     fix(a_or[i], obs.line_or_bus[i] - 1)
    #     fix(a_ex[i], obs.line_ex_bus[i] - 1)
    #
    # # After fixing, print how many are actually fixed
    # fixed = sum(1 for i in range(n_bus) if Vm[i].LOWER == Vm[i].UPPER)
    # print(f"Fixed Vm: {fixed} / {n_bus}")
    #
    # m.options.SOLVER = 1  # APOPT (needed for integer vars)
    # m.options.IMODE = 3  # steady-state optimization
    #
    # m.Minimize(0)  # no objective, just check constraints
    # m.solve(disp=True)
    #
    #
    #
    # # Print intermediate values to verify balance equations
    # print("Checking power balance residuals...")
    # for bus_id, P_res, Q_res in residuals:
    #     print(f"bus {bus_id}: P_res={P_res.value[0]:.6f}, Q_res={Q_res.value[0]:.6f}")


if __name__ == "__main__":
    main()
