import grid2op
import numpy as np
from gekko import GEKKO
from gekko.gk_variable import GKVariable
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
    def __init__(
        self, env: Environment, obs: Observation, utilization_threshold: float, validation_mode: bool = False
    ) -> None:
        self.env = env
        self.obs = obs
        self.validation_mode = validation_mode
        self.utilization_threshold = utilization_threshold

        self.m = GEKKO(remote=False)

        self.n_sub, self.n_bus, self.n_gen, self.n_load, self.n_line = get_grid_sizes(env)

        if validation_mode:
            self.Vm = self.m.Array(self.m.Var, self.n_bus, lb=0, value=1)
            self.theta = self.m.Array(self.m.Var, self.n_bus, lb=-np.pi, ub=np.pi)
            self.Pg = self.m.Array(self.m.Var, self.n_gen, value=0)
            self.Qg = self.m.Array(self.m.Var, self.n_gen, value=0)
            self.a_gen = self.m.Array(self.m.Param, self.n_gen, value=0)
            self.a_load = self.m.Array(self.m.Param, self.n_load, value=0)
            self.a_or = self.m.Array(self.m.Param, self.n_line, value=0)
            self.a_ex = self.m.Array(self.m.Param, self.n_line, value=0)
        else:
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
        self.line_f_indices = set()
        self.utilizations = {}

        self.debug_Vm_res = {}
        self.debug_theta_res = {}
        self.debug_pf_line = {}
        self.debug_qf_line = {}

    def fix(self, var, value):
        var.value = value
        if isinstance(var, GKVariable):
            var.LOWER = value
            var.UPPER = value

    def add_bus_type_constraints(self):
        print("About to add bus type constraints")
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

                # self.m.Equation(self.Vm[bus_id] >= 0.0)

                self.m.Equation(self.Vm[bus_id] >= Vm_res - self.M_vm * a_gs)
                self.m.Equation(self.Vm[bus_id] <= Vm_res + self.M_vm * a_gs)

                self.m.Equation(self.theta[bus_id] >= theta_res - self.M_th * a_gs)
                self.m.Equation(self.theta[bus_id] <= theta_res + self.M_th * a_gs)

                continue

            if len(gen_ids) > 0:
                # bus is either PV or PQ based on if there are any generators connected to it

                # fix all the generator active powers, because they're gonna be attached to PV buses anyhow
                for g in gen_ids:
                    self.fix(self.Pg[g], self.net.res_gen.p_mw[g] / self.baseMVA)

                # Qg remains free
                # theta remains free

                # Vm is fixed if there's at least one generator connected to the busbar

                if busbar == 1:
                    a_g = self.a_gen[gen_ids]
                else:
                    a_g = 1 - self.a_gen[gen_ids]

                n_g = len(a_g)

                self.debug_Vm_res[bus_id] = Vm_res
                self.debug_theta_res[bus_id] = theta_res

                z = self.m.Var(integer=True, lb=0, ub=1)  # z == 0 if at least one generator is at busbar
                for a_gi in a_g:
                    self.m.Equation(z <= a_gi)

                self.m.Equation(z >= self.m.sum(a_g) - (n_g - 1))

                self.m.Equation(self.Vm[bus_id] >= Vm_res - self.M_vm * z)
                self.m.Equation(self.Vm[bus_id] <= Vm_res + self.M_vm * z)

    def add_power_flow_equations(self):
        for bus_id in range(self.n_bus):
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

                Vm_f = self.Vm[bus_id]
                theta_f = self.theta[bus_id]

                Vm_t1 = self.Vm[to_buses[0]]
                theta_t1 = self.theta[to_buses[0]]

                Vm_t2 = self.Vm[to_buses[1]]
                theta_t2 = self.theta[to_buses[1]]

                # fmt: off
                if busbar == 1:
                    Pf_line = Vm_f*(1 - a_f)*(Vm_f*(1 - a_f)*(-Yff_i*self.m.sin(theta_f) + Yff_r*self.m.cos(theta_f))*self.m.cos(theta_f) - Vm_f*(1 - a_f)*(-Yff_i*self.m.cos(theta_f) - Yff_r*self.m.sin(theta_f))*self.m.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*self.m.cos(theta_t1) - Vm_t2*Yft_i*a_t*self.m.sin(theta_t2) + Vm_t2*Yft_r*a_t*self.m.cos(theta_t2))*self.m.cos(theta_f) - (-Vm_t1*Yft_i*(1 - a_t)*self.m.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*self.m.sin(theta_t1) - Vm_t2*Yft_i*a_t*self.m.cos(theta_t2) - Vm_t2*Yft_r*a_t*self.m.sin(theta_t2))*self.m.sin(theta_f))  # noqa: E226
                    Qf_line = Vm_f*(1 - a_f)*(Vm_f*(1 - a_f)*(-Yff_i*self.m.sin(theta_f) + Yff_r*self.m.cos(theta_f))*self.m.sin(theta_f) + Vm_f*(1 - a_f)*(-Yff_i*self.m.cos(theta_f) - Yff_r*self.m.sin(theta_f))*self.m.cos(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*self.m.cos(theta_t1) - Vm_t2*Yft_i*a_t*self.m.sin(theta_t2) + Vm_t2*Yft_r*a_t*self.m.cos(theta_t2))*self.m.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*self.m.sin(theta_t1) - Vm_t2*Yft_i*a_t*self.m.cos(theta_t2) - Vm_t2*Yft_r*a_t*self.m.sin(theta_t2))*self.m.cos(theta_f))  # noqa: E226
                else:
                    Pf_line = Vm_f*a_f*(Vm_f*a_f*(-Yff_i*self.m.sin(theta_f) + Yff_r*self.m.cos(theta_f))*self.m.cos(theta_f) - Vm_f*a_f*(-Yff_i*self.m.cos(theta_f) - Yff_r*self.m.sin(theta_f))*self.m.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*self.m.cos(theta_t1) - Vm_t2*Yft_i*a_t*self.m.sin(theta_t2) + Vm_t2*Yft_r*a_t*self.m.cos(theta_t2))*self.m.cos(theta_f) - (-Vm_t1*Yft_i*(1 - a_t)*self.m.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*self.m.sin(theta_t1) - Vm_t2*Yft_i*a_t*self.m.cos(theta_t2) - Vm_t2*Yft_r*a_t*self.m.sin(theta_t2))*self.m.sin(theta_f))  # noqa: E226
                    Qf_line = Vm_f*a_f*(Vm_f*a_f*(-Yff_i*self.m.sin(theta_f) + Yff_r*self.m.cos(theta_f))*self.m.sin(theta_f) + Vm_f*a_f*(-Yff_i*self.m.cos(theta_f) - Yff_r*self.m.sin(theta_f))*self.m.cos(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.sin(theta_t1) + Vm_t1*Yft_r*(1 - a_t)*self.m.cos(theta_t1) - Vm_t2*Yft_i*a_t*self.m.sin(theta_t2) + Vm_t2*Yft_r*a_t*self.m.cos(theta_t2))*self.m.sin(theta_f) + (-Vm_t1*Yft_i*(1 - a_t)*self.m.cos(theta_t1) - Vm_t1*Yft_r*(1 - a_t)*self.m.sin(theta_t1) - Vm_t2*Yft_i*a_t*self.m.cos(theta_t2) - Vm_t2*Yft_r*a_t*self.m.sin(theta_t2))*self.m.cos(theta_f))  # noqa: E226
                # fmt: on

                Pf_total = Pf_total + Pf_line
                Qf_total = Qf_total + Qf_line

                if (bus_id, line_idx) not in self.line_f_indices:
                    self.line_f_indices.add((bus_id, line_idx))

                    S_MVA = self.m.sqrt(Pf_line**2 + Qf_line**2) * self.baseMVA
                    vn_bus_kv = self.net.bus.loc[bus_id].vn_kv
                    If_A = S_MVA * 1e6 / (np.sqrt(3) * vn_bus_kv * Vm_f * 1e3)

                    utilization = self.m.Intermediate(If_A / self.obs.thermal_limit[line_idx])
                    self.utilizations[(bus_id, line_idx)] = utilization

                    self.m.Equation(utilization < self.utilization_threshold)

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

                Vm_t = self.Vm[bus_id]
                theta_t = self.theta[bus_id]

                Vm_f1 = self.Vm[from_buses[0]]
                theta_f1 = self.theta[from_buses[0]]

                Vm_f2 = self.Vm[from_buses[1]]
                theta_f2 = self.theta[from_buses[1]]

                # fmt: off
                if busbar == 1:
                    Pt_line = Vm_t*(1 - a_t)*(Vm_t*(1 - a_t)*(-Ytt_i*self.m.sin(theta_t) + Ytt_r*self.m.cos(theta_t))*self.m.cos(theta_t) - Vm_t*(1 - a_t)*(-Ytt_i*self.m.cos(theta_t) - Ytt_r*self.m.sin(theta_t))*self.m.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*self.m.cos(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.sin(theta_f2) + Vm_f2*Ytf_r*a_f*self.m.cos(theta_f2))*self.m.cos(theta_t) - (-Vm_f1*Ytf_i*(1 - a_f)*self.m.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*self.m.sin(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.cos(theta_f2) - Vm_f2*Ytf_r*a_f*self.m.sin(theta_f2))*self.m.sin(theta_t))  # noqa: E226
                    Qt_line = Vm_t*(1 - a_t)*(Vm_t*(1 - a_t)*(-Ytt_i*self.m.sin(theta_t) + Ytt_r*self.m.cos(theta_t))*self.m.sin(theta_t) + Vm_t*(1 - a_t)*(-Ytt_i*self.m.cos(theta_t) - Ytt_r*self.m.sin(theta_t))*self.m.cos(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*self.m.cos(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.sin(theta_f2) + Vm_f2*Ytf_r*a_f*self.m.cos(theta_f2))*self.m.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*self.m.sin(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.cos(theta_f2) - Vm_f2*Ytf_r*a_f*self.m.sin(theta_f2))*self.m.cos(theta_t))  # noqa: E226
                else:
                    Pt_line = Vm_t*a_t*(Vm_t*a_t*(-Ytt_i*self.m.sin(theta_t) + Ytt_r*self.m.cos(theta_t))*self.m.cos(theta_t) - Vm_t*a_t*(-Ytt_i*self.m.cos(theta_t) - Ytt_r*self.m.sin(theta_t))*self.m.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*self.m.cos(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.sin(theta_f2) + Vm_f2*Ytf_r*a_f*self.m.cos(theta_f2))*self.m.cos(theta_t) - (-Vm_f1*Ytf_i*(1 - a_f)*self.m.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*self.m.sin(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.cos(theta_f2) - Vm_f2*Ytf_r*a_f*self.m.sin(theta_f2))*self.m.sin(theta_t))  # noqa: E226
                    Qt_line = Vm_t*a_t*(Vm_t*a_t*(-Ytt_i*self.m.sin(theta_t) + Ytt_r*self.m.cos(theta_t))*self.m.sin(theta_t) + Vm_t*a_t*(-Ytt_i*self.m.cos(theta_t) - Ytt_r*self.m.sin(theta_t))*self.m.cos(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.sin(theta_f1) + Vm_f1*Ytf_r*(1 - a_f)*self.m.cos(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.sin(theta_f2) + Vm_f2*Ytf_r*a_f*self.m.cos(theta_f2))*self.m.sin(theta_t) + (-Vm_f1*Ytf_i*(1 - a_f)*self.m.cos(theta_f1) - Vm_f1*Ytf_r*(1 - a_f)*self.m.sin(theta_f1) - Vm_f2*Ytf_i*a_f*self.m.cos(theta_f2) - Vm_f2*Ytf_r*a_f*self.m.sin(theta_f2))*self.m.cos(theta_t))  # noqa: E226
                # fmt: on

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

    problem = MINLP(env, obs, utilization_threshold=1.0)
    problem.add_bus_type_constraints()
    problem.add_power_flow_equations()

    # TODO extract the action dict from the solutions

    # TODO finally, run an episode while solving the problem
    #  check if the line utilizations are really bellow the threshold

    # Fix voltage variables
    for i in range(problem.n_bus):
        if np.isnan(problem.net.res_bus.vm_pu[i]):
            # Mirror from the corresponding busbar 1
            busbar1_id = i - problem.n_sub  # since busbar 2 = busbar 1 index + n_sub
            problem.fix(problem.Vm[i], problem.net.res_bus.vm_pu[busbar1_id])
            problem.fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[busbar1_id]))
        else:
            problem.fix(problem.Vm[i], problem.net.res_bus.vm_pu[i])
            problem.fix(problem.theta[i], np.deg2rad(problem.net.res_bus.va_degree[i]))

    for i in range(problem.n_gen):
        problem.fix(problem.Pg[i], problem.net.res_gen.p_mw[i] / problem.baseMVA)
        problem.fix(problem.Qg[i], problem.net.res_gen.q_mvar[i] / problem.baseMVA)

    # Fix binary switching variables
    for i in range(problem.n_gen):
        problem.fix(problem.a_gen[i], obs.gen_bus[i] - 1)

    for i in range(problem.n_load):
        problem.fix(problem.a_load[i], obs.load_bus[i] - 1)

    for i in range(problem.n_line):
        problem.fix(problem.a_or[i], obs.line_or_bus[i] - 1)
        problem.fix(problem.a_ex[i], obs.line_ex_bus[i] - 1)

    problem.m.options.SOLVER = 1  # APOPT (needed for integer vars)
    problem.m.options.IMODE = 3  # steady-state optimization

    problem.m.Minimize(0)  # no objective, just check constraints
    problem.m.solve(disp=True)

    print("After solving")
    for bus_id in problem.debug_Vm_res:
        Vm = problem.Vm[bus_id].value[0]
        theta = problem.theta[bus_id].value[0]

        Vm_res = problem.debug_Vm_res[bus_id]
        theta_res = problem.debug_theta_res[bus_id]

        print(f"{bus_id=}, {Vm=}, {theta=}")
        print(f"{bus_id=}, {Vm_res=}, {theta_res=}")

        assert np.isclose(Vm, Vm_res)
        assert np.isclose(theta, theta_res)

    # Print intermediate values to verify balance equations
    print("Checking power balance residuals...")
    for bus_id, P_res, Q_res in problem.residuals:
        print(f"bus {bus_id}: P_res={P_res.value[0]:.6f}, Q_res={Q_res.value[0]:.6f}")
        assert P_res.value[0] < 1e-6
        assert Q_res.value[0] < 1e-6


if __name__ == "__main__":
    main()
