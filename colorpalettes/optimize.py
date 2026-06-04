import itertools
from dataclasses import dataclass, field
from typing import Callable, Collection, Iterable

import numpy as np
import scipy.optimize

from colorpalettes.color import Color, ColorDeficiencyType
from colorpalettes.colorset import Colorset

Potential = Callable[[float], float]


def make_coulomb_potential(lambda_: float, n: float = 1) -> Potential:
    def coulomb(d: float) -> float:
        return (lambda_ / d) ** n

    return coulomb


def make_LJ_potential(lambda_: float, n: float = 12, m: float = 6) -> Potential:
    def lj(d: float) -> float:
        return (lambda_ / d) ** n - (lambda_ / d) ** m

    return lj


@dataclass
class BackgroundColor:
    color: Color
    weight: float


@dataclass
class ColorsetOptimizationConfig:
    backgrounds: list[BackgroundColor] | None = field(default_factory=lambda: [BackgroundColor(Color.white(), 1.0)])
    standard_color_weight: float = 1.0
    colorblind_colors_weight: float = 1.0
    global_constraints_weight: float = 1.0

    rgb_border_thickness: float = 1 / 255  # thickness of ~1 bit in the RGB space for the RGB validity potential

    freeze_first: bool = False

    # float is interpreted as a sigma in Delta E for the standard Lennard-Jones potential
    potential: Potential | float = 30.0

    verbose: bool = False
    minimization_algorithm: str = "L-BFGS-B"
    minimization_algorithm_options: dict | None = None

    hue_valley: tuple[float, float] | None = None
    chroma_valley: tuple[float, float] | None = None
    lightness_valley: tuple[float, float] | None = None


def optimize_colorset(init_colorset: Colorset, config: ColorsetOptimizationConfig) -> Colorset:
    first = init_colorset.colors[0]
    if callable(config.potential):
        potential = config.potential
    else:
        potential = make_LJ_potential(config.potential)

    def pack_color(c: Color) -> Iterable[float]:
        return c.JCh

    def unpack_color(vec: Collection[float], is_output: bool = False) -> Color:
        return Color.from_JCh(vec, ensure_valid=is_output, validate=is_output)  # type: ignore

    def pack_set(cs: Colorset) -> np.ndarray:
        start_idx = 1 if config.freeze_first else 0
        return np.array(list(itertools.chain.from_iterable(pack_color(c) for c in cs.colors[start_idx:])))

    def unpack_set(vec: np.ndarray, is_output: bool = False) -> Colorset:
        colors = [unpack_color(coords, is_output=is_output) for coords in itertools.batched(vec, n=3)]
        if config.freeze_first:
            colors.insert(0, first)
        return Colorset(tuple(colors))

    def pairwise_cost(colorset: Colorset) -> float:
        total = 0
        for i, row in enumerate(colorset.distance_mat):
            for j in range(i + 1, len(row)):
                if row[j] < 1e-8:
                    continue
                total += potential(row[j])
        return total / colorset.n

    def loss(vec: np.ndarray) -> float:
        cs = unpack_set(vec)

        # global colors potential
        global_loss = 0

        # RGB validity potential
        for c in cs.colors:
            lambda_ = config.rgb_border_thickness
            for v in c.rgb:
                global_loss += np.exp(-v / lambda_) + np.exp((v - 1) / lambda_)

        # valley potentials for lightness, chroma, and hue
        if config.lightness_valley or config.hue_valley or config.chroma_valley:
            for color in cs.colors:
                J, C, hue = color.JCh
                if config.lightness_valley:
                    J_mean, J_std = config.lightness_valley
                    global_loss += ((J - J_mean) / J_std) ** 2 / 2
                if config.chroma_valley:
                    C_mean, C_std = config.chroma_valley
                    global_loss += ((C - C_mean) / C_std) ** 2 / 2
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

        # color must be distinct from the background
        if config.backgrounds is not None:
            for bg in config.backgrounds:
                for c in cs.colors:
                    global_loss += bg.weight * potential(c.delta_E(bg.color))

        standard_pairwise_loss = pairwise_cost(cs)

        colorblind_pairwise_loss = 0
        if config.colorblind_colors_weight > 0:
            for cb_type in ColorDeficiencyType:
                colorblind_pairwise_loss += pairwise_cost(cs.color_deficient(cb_type))
            colorblind_pairwise_loss /= len(ColorDeficiencyType)

        return (
            config.global_constraints_weight * global_loss
            + config.standard_color_weight * standard_pairwise_loss
            + config.colorblind_colors_weight * colorblind_pairwise_loss
        )

    res = scipy.optimize.minimize(
        loss,
        x0=pack_set(init_colorset),
        method=config.minimization_algorithm,
        options=config.minimization_algorithm_options,
    )
    if config.verbose:
        print(res)
    return unpack_set(res.x, is_output=True)
