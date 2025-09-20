import hydra
import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO

from src import CONFIGS_PATH
from src.config.config import Config
from src.plot.utils import set_rcParams


@hydra.main(version_base=None, config_path=str(CONFIGS_PATH), config_name="default")
def main(cfg: Config):
    """
    An example script.

    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------
    """

    set_rcParams(cfg)

    x = np.linspace(-2, 2, 100)

    fig, axs = plt.subplots(1, 1)
    axs.plot(x, x**2)
    axs.set_xlabel("x")
    axs.set_ylabel("y")

    m = GEKKO(remote=False)  # create GEKKO model
    x = m.Var(lb=0.5, integer=True)  # define new variable, default=0
    m.Equation(x >= 1.1)
    m.Minimize(x**2)

    m.solve(disp=False)  # solve
    obj_opt = m.options.objfcnval
    print(f"{x.value=}\n{obj_opt=}")  # print solution

    axs.scatter(x.value[0], obj_opt, c="r", s=100)

    m = GEKKO(remote=False)  # create GEKKO model

    x = m.Array(m.Var, 2)
    x[0].lower = 1.0
    m.Minimize(x.T @ x)

    m.fix(x[1], val=0.3)

    m.solve(disp=False)  # solve
    obj_opt = m.options.objfcnval
    print(f"\n\n{x[0].value=}\n{x[1].value=}\n{obj_opt=}")  # print solution

    m = GEKKO(remote=False)  # create GEKKO model
    x = m.Var(lb=0.5, integer=True)  # define new variable, default=0

    eq = x >= 1.1
    m.Equation(eq)
    m.Minimize(m.exp(x))

    m.solve(disp=False)  # solve
    obj_opt = m.options.objfcnval
    print(f"{x.value=}\n{obj_opt=}")  # print solution

    m = GEKKO(remote=False)  # create GEKKO model
    x = m.Var()  # define new variable, default=0

    left = x - 2
    right = 2 * x + 1
    m.Equation(left == right)
    m.solve(disp=False)  # solve

    # m = GEKKO(remote=False)  # create GEKKO model
    # x = m.Var()
    # y = m.Var()
    # left = x * m.exp(1j * y) + 2 + 1j*3.3
    # right = 4 + 1.j * 0.1
    # m.Equation(x * m.exp(1j * y) + 2 + 1j*3.3 == 4 + 1.j * 0.1)
    # m.solve(disp=False)  # solve

    plt.show()


if __name__ == "__main__":
    main()
