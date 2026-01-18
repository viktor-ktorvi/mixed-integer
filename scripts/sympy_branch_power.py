import sympy as sp
from sympy.printing.numpy import NumPyPrinter


def polar(mag, angle):
    return mag * sp.exp(sp.I * angle)


def cartesian(real, imag):
    return real + sp.I * imag


def to_numpy_equation(expression, printer) -> str:
    return printer.doprint(expression).replace("numpy", "np")


def get_real_imag(expression):
    return sp.re(expression), sp.im(expression)


def main_f():
    print("\n====== FROM ======\n")
    a_f, a_t = sp.symbols("a_f, a_t", real=True)
    V_f, V_t1, V_t2 = sp.symbols("V_f, V_t1, V_t2", complex=True)
    Yff, Yft = sp.symbols("Yff, Yft", complex=True)

    I_f1 = (1 - a_f) * V_f * Yff + ((1 - a_t) * V_t1 + a_t * V_t2) * Yft
    S_f1 = (1 - a_f) * V_f * sp.conjugate(I_f1)

    # TODO mislim da moze bez ovog a sto mnozi Vf, jer ce sve ionako biti 0 kod Sf
    I_f2 = a_f * V_f * Yff + ((1 - a_t) * V_t1 + a_t * V_t2) * Yft
    S_f2 = a_f * V_f * sp.conjugate(I_f2)

    Vm_f, theta_f = sp.symbols("Vm_f, theta_f", real=True)
    Vm_t1, theta_t1 = sp.symbols("Vm_t1, theta_t1", real=True)
    Vm_t2, theta_t2 = sp.symbols("Vm_t2, theta_t2", real=True)

    Yff_r, Yff_i = sp.symbols("Yff_r, Yff_i", real=True)
    Yft_r, Yft_i = sp.symbols("Yft_r, Yft_i", real=True)

    S_f1_subs = S_f1.subs(
        {
            V_f: polar(Vm_f, theta_f),
            V_t1: polar(Vm_t1, theta_t1),
            V_t2: polar(Vm_t2, theta_t2),
            Yff: cartesian(Yff_r, Yff_i),
            Yft: cartesian(Yft_r, Yft_i),
        }
    ).rewrite(sp.cos)

    S_f2_subs = S_f2.subs(
        {
            V_f: polar(Vm_f, theta_f),
            V_t1: polar(Vm_t1, theta_t1),
            V_t2: polar(Vm_t2, theta_t2),
            Yff: cartesian(Yff_r, Yff_i),
            Yft: cartesian(Yft_r, Yft_i),
        }
    ).rewrite(sp.cos)

    P_f1, Q_f1 = get_real_imag(S_f1_subs)
    P_f2, Q_f2 = get_real_imag(S_f2_subs)

    printer = NumPyPrinter()
    print(f"P_f1 = {to_numpy_equation(P_f1, printer)}  # noqa: E226")
    print(f"Q_f1 = {to_numpy_equation(Q_f1, printer)}  # noqa: E226\n\n")

    print(f"P_f2 = {to_numpy_equation(P_f2, printer)}  # noqa: E226")
    print(f"Q_f2 = {to_numpy_equation(Q_f2, printer)}  # noqa: E226")


def main_t():
    print("\n====== TO ======\n")

    a_f, a_t = sp.symbols("a_f, a_t", real=True)

    V_t, V_f1, V_f2 = sp.symbols("V_t, V_f1, V_f2", complex=True)
    Ytf, Ytt = sp.symbols("Ytf, Ytt", complex=True)

    I_t1 = ((1 - a_f) * V_f1 + a_f * V_f2) * Ytf + (1 - a_t) * V_t * Ytt
    S_t1 = (1 - a_t) * V_t * sp.conjugate(I_t1)

    I_t2 = ((1 - a_f) * V_f1 + a_f * V_f2) * Ytf + a_t * V_t * Ytt
    S_t2 = a_t * V_t * sp.conjugate(I_t2)

    Vm_t, theta_t = sp.symbols("Vm_t, theta_t", real=True)
    Vm_f1, theta_f1 = sp.symbols("Vm_f1, theta_f1", real=True)
    Vm_f2, theta_f2 = sp.symbols("Vm_f2, theta_f2", real=True)

    Ytf_r, Ytf_i = sp.symbols("Ytf_r, Ytf_i", real=True)
    Ytt_r, Ytt_i = sp.symbols("Ytt_r, Ytt_i", real=True)

    S_t1_subs = S_t1.subs(
        {
            V_t: polar(Vm_t, theta_t),
            V_f1: polar(Vm_f1, theta_f1),
            V_f2: polar(Vm_f2, theta_f2),
            Ytf: cartesian(Ytf_r, Ytf_i),
            Ytt: cartesian(Ytt_r, Ytt_i),
        }
    ).rewrite(sp.cos)

    S_t2_subs = S_t2.subs(
        {
            V_t: polar(Vm_t, theta_t),
            V_f1: polar(Vm_f1, theta_f1),
            V_f2: polar(Vm_f2, theta_f2),
            Ytf: cartesian(Ytf_r, Ytf_i),
            Ytt: cartesian(Ytt_r, Ytt_i),
        }
    ).rewrite(sp.cos)

    P_t1, Q_t1 = get_real_imag(S_t1_subs)
    P_t2, Q_t2 = get_real_imag(S_t2_subs)

    printer = NumPyPrinter()
    print(f"P_t1 = {to_numpy_equation(P_t1, printer)}  # noqa: E226")
    print(f"Q_t1 = {to_numpy_equation(Q_t1, printer)}  # noqa: E226\n\n")

    print(f"P_t2 = {to_numpy_equation(P_t2, printer)}  # noqa: E226")
    print(f"Q_t2 = {to_numpy_equation(Q_t2, printer)}  # noqa: E226")


if __name__ == "__main__":
    main_f()
    print()
    main_t()
