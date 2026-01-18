import grid2op
from grid2op.PlotGrid import PlotMatplot


def main():
    env_name = "l2rpn_case14_sandbox"  # for example, other environments might be usable
    # env_name = "l2rpn_icaps_2021_small"
    env = grid2op.make(env_name)

    obs = env.reset()
    plot_helper = PlotMatplot(env.observation_space, line_id=True, gen_id=True, load_id=True)
    fig = plot_helper.plot_obs(obs)
    fig.show()


if __name__ == "__main__":
    main()
