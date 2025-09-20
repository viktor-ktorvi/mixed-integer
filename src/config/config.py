from dataclasses import dataclass, field

from src.config.plotting.plotting import Plotting


@dataclass
class Config:
    random_seed: int
    plotting: Plotting = field(default_factory=Plotting)
