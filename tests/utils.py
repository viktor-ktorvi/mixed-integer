from types import SimpleNamespace

import numpy as np

from scripts.minlp import MINLP


def make_obs_from_gekko(problem: MINLP) -> SimpleNamespace:
    baseMVA = problem.baseMVA

    return SimpleNamespace(
        gen_p=np.array([problem.Pg[i].value[0] for i in range(problem.n_gen)]) * baseMVA,
        gen_q=np.array([problem.Qg[i].value[0] for i in range(problem.n_gen)]) * baseMVA,
        gen_bus=np.array([int(problem.a_gen[i].value[0]) + 1 for i in range(problem.n_gen)]),
        load_bus=np.array([int(problem.a_load[i].value[0]) + 1 for i in range(problem.n_load)]),
        line_or_bus=np.array([int(problem.a_or[i].value[0]) + 1 for i in range(problem.n_line)]),
        line_ex_bus=np.array([int(problem.a_ex[i].value[0]) + 1 for i in range(problem.n_line)]),
        load_p=problem.obs.load_p,
        load_q=problem.obs.load_q,
        p_or=problem.obs.p_or,
        q_or=problem.obs.q_or,
        p_ex=problem.obs.p_ex,
        q_ex=problem.obs.q_ex,
        line_or_to_subid=problem.obs.line_or_to_subid,
        line_ex_to_subid=problem.obs.line_ex_to_subid,
        thermal_limit=problem.obs.thermal_limit,
    )
