from dataclasses import dataclass, field


@dataclass
class Axes:
    grid: bool = True


@dataclass
class Figure:
    autolayout: bool = True
    figsize: tuple[float, float] = (16, 9)
    dpi: float = 130.0


@dataclass
class Font:
    size: int = 14


@dataclass
class Plotting:
    axes: Axes = field(default_factory=Axes)
    figure: Figure = field(default_factory=Figure)
    font: Font = field(default_factory=Font)
    subplot_title_fontsize: int = 13
