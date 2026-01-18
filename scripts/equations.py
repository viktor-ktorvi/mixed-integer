import grid2op

from src.power_flow.validate_equations import validate_equations


def main():
    # env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name)
    obs = env.reset()

    validate_equations(env, obs, threshold=1e-4)


if __name__ == "__main__":
    main()
