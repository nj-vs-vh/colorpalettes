import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.optimize

from colorpalettes.color import Color, ColorblindnessType
from colorpalettes.colorset import Colorset

Potential = Callable[[float], float]


def make_coulomb_potential(d_1: float) -> Potential:
    def coulomb(d: float) -> float:
        return d_1 / d

    return coulomb


def make_LJ_potential(sigma: float, n: float = 12, m: float = 6) -> Potential:
    def coulomb(d: float) -> float:
        return (sigma / d) ** n - (sigma / d) ** m

    return coulomb


@dataclass
class ColorsetOptimizationConfig:
    freeze_first: bool = False
    standard_color_weight: float = 1.0
    colorblind_colors_weight: float = 1.0
    global_constraints_weight: float = 1.0
    hue_valley: tuple[float, float] | None = None
    saturation_valley: tuple[float, float] = (60, 20)
    lightness_valley: tuple[float, float] = (70, 20)
    background: Color | None = field(default_factory=Color.white)
    minimization_algorithm: str = "L-BFGS-B"
    potential: Potential = make_LJ_potential(30)
    verbose: bool = False


def optimize_colorset(init_colorset: Colorset, config: ColorsetOptimizationConfig) -> Colorset:
    init_colorset = Colorset(tuple(c.not_extreme() for c in init_colorset))
    first = init_colorset.colors[0]

    def unpack(hsl_vec: np.ndarray) -> Colorset:
        colors = [Color.from_hsl(tuple(hsl)) for hsl in itertools.batched(hsl_vec, n=3)]
        if config.freeze_first:
            colors.insert(0, first)

        return Colorset(tuple(colors))

    def pairwise_cost(colorset: Colorset) -> float:
        total = 0
        for i, row in enumerate(colorset.distance_mat):
            for j in range(i + 1, len(row)):
                if row[j] < 1e-8:
                    continue
                total += config.potential(row[j])
        return total / colorset.n

    def objective(hsl_vec: np.ndarray) -> float:
        cs = unpack(hsl_vec)

        # global colors potential
        global_loss = 0
        # extreme colors dampin
        delta = 1
        power = 4
        for color in cs.colors:
            for val in (color.hsl[1], color.hsl[2]):
                if val <= 0.0 or val >= 100.0:
                    return np.inf
                global_loss += (val / delta) ** -power + ((100 - val) / delta) ** -power
        # sat and lght valleys
        sat_mean, sat_std = config.saturation_valley
        lght_mean, lght_std = config.lightness_valley
        for color in cs.colors:
            hue, sat, lght = color.hsl
            global_loss += ((lght - lght_mean) / lght_std) ** 2 / 2 + ((sat - sat_mean) / sat_std) ** 2 / 2
            if config.hue_valley:
                hue_mean, hue_std = config.hue_valley
                hue_residual = min(  # accounting for hue wrapping over 360
                    np.abs(
                        [
                            hue - hue_mean,
                            hue - 360 - hue_mean,
                            hue + 360 - hue_mean,
                        ]
                    )
                )
                global_loss += (hue_residual / hue_std) ** 2 / 2
        # background color distinctness
        if config.background is not None:
            for c in cs:
                global_loss += 30 / c.delta_E(config.background)

        standard_pairwise_loss = pairwise_cost(cs)

        colorblind_pairwise_loss = 0
        for cb_type in ColorblindnessType:
            colorblind_pairwise_loss += pairwise_cost(cs.colorblinded(cb_type))
        colorblind_pairwise_loss /= len(ColorblindnessType)

        return (
            config.global_constraints_weight * global_loss
            + config.standard_color_weight * standard_pairwise_loss
            + config.colorblind_colors_weight * colorblind_pairwise_loss
        )

    x0 = np.array(
        list(itertools.chain.from_iterable(c.hsl for c in init_colorset.colors[(1 if config.freeze_first else 0) :]))
    )

    res = scipy.optimize.minimize(
        objective,
        x0=x0,
        method=config.minimization_algorithm,
    )
    if config.verbose:
        print(res)
    return unpack(res.x)
