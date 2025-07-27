import numpy as np

from colorpalettes.color import Color
from colorpalettes.colorset import Colorset


def random_colorset(n_colors: int, random_seed: int | None = None) -> Colorset:
    rng = np.random.default_rng(seed=random_seed)
    hues = rng.uniform(low=0, high=360, size=n_colors) % 360
    sats = np.clip(rng.normal(loc=50, scale=30, size=n_colors), 0.1, 99.9)
    lghts = np.clip(rng.normal(loc=50, scale=30, size=n_colors), 0.1, 99.9)
    return Colorset(tuple(Color.from_hsl(hsl) for hsl in zip(hues, sats, lghts)))


def random_complimentary_colorset(
    start: Color,
    n_colors: int,
    n_color_cycle: int = 3,
    hue_step_std: float = 10,
    sat_std: float = 30,
    lightness_std: float = 10,
    random_seed: int | None = None,
) -> Colorset:
    rng = np.random.default_rng(seed=random_seed)

    hue_start, sat_start, lght_start = start.hsl

    mean_hue_step = 360 / n_color_cycle
    hue_steps = mean_hue_step * rng.normal(loc=1, scale=hue_step_std / mean_hue_step, size=n_colors - 1)
    hues = np.concatenate(([hue_start], (hue_start + np.cumsum(hue_steps)) % 360))

    sats = np.clip(rng.normal(loc=sat_start, scale=sat_std, size=n_colors), 0, 100)
    sats[0] = sat_start
    lghts = np.clip(rng.normal(loc=lght_start, scale=lightness_std, size=n_colors), 0, 100)
    lghts[0] = lght_start
    return Colorset(tuple(Color.from_hsl(hsl) for hsl in zip(hues, sats, lghts)))
