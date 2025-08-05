import numpy as np

from colorpalettes.color import Color
from colorpalettes.colorset import Colorset


def random_colorset(n_colors: int, random_seed: int | None = None) -> Colorset:
    rng = np.random.default_rng(seed=random_seed)
    Js = rng.normal(loc=50, scale=20, size=n_colors)
    Cs = rng.normal(loc=50, scale=20, size=n_colors)
    hues = rng.uniform(low=0, high=360, size=n_colors) % 360
    return Colorset(tuple(Color.from_JCh(JCh, ensure_valid=True) for JCh in zip(Js, Cs, hues)))


def random_complimentary_colorset(
    start: Color,
    n_colors: int,
    n_color_cycle: int = 3,
    hue_step_std: float = 10,
    C_std: float = 30,
    J_std: float = 10,
    random_seed: int | None = None,
) -> Colorset:
    rng = np.random.default_rng(seed=random_seed)

    J_start, C_start, h_start = start.JCh

    mean_hue_step = 360 / n_color_cycle
    hue_steps = mean_hue_step * rng.normal(loc=1, scale=hue_step_std / mean_hue_step, size=n_colors - 1)
    hues = np.concatenate(([h_start], (h_start + np.cumsum(hue_steps)) % 360))

    Cs = np.clip(rng.normal(loc=C_start, scale=C_std, size=n_colors), 0, 100)
    Cs[0] = C_start
    Js = np.clip(rng.normal(loc=J_start, scale=J_std, size=n_colors), 0, 100)
    Js[0] = J_start
    return Colorset(tuple(Color.from_JCh(JCh, ensure_valid=True) for JCh in zip(Js, Cs, hues)))
