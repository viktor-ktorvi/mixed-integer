import hydra
from omegaconf import DictConfig

from src import CONFIGS_PATH


@hydra.main(version_base=None, config_path=str(CONFIGS_PATH), config_name="default")
def main(cfg: DictConfig):
    """
    An example script.

    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------
    """
    print("Hello world")
    print(f"{cfg.random_seed=}")


if __name__ == "__main__":
    main()
