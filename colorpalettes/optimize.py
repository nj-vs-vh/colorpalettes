import itertools
from dataclasses import dataclass, field
from typing import Callable, Collection, Iterable

import numpy as np
import scipy.optimize

from colorpalettes.color import Color, ColorDeficiencyType
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
    chroma_valley: tuple[float, float] = (60, 20)
    lightness_valley: tuple[float, float] = (70, 20)
    background: Color | None = field(default_factory=Color.white)
    minimization_algorithm: str = "L-BFGS-B"
    potential: Potential = make_LJ_potential(30)
    verbose: bool = False


def optimize_colorset(init_colorset: Colorset, config: ColorsetOptimizationConfig) -> Colorset:
    first = init_colorset.colors[0]

    def encode(c: Color) -> Iterable[float]:
        return c.hsv

    def decode(vec: Collection[float]) -> Color:
        h, s, v = vec
        h = h % 1.0
        return Color.from_hsv((h, s, v), ensure_valid=False)  # type: ignore

    def pack(cs: Colorset) -> np.ndarray:
        start_idx = 1 if config.freeze_first else 0
        return np.array(list(itertools.chain.from_iterable(encode(c) for c in cs.colors[start_idx:])))

    def unpack(vec: np.ndarray) -> Colorset | None:
        try:
            colors = [decode(coords) for coords in itertools.batched(vec, n=3)]
        except ValueError:
            return None

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

    def loss(vec: np.ndarray) -> float:
        cs = unpack(vec)
        if cs is None:
            return 1e10

        # global colors potential
        global_loss = 0

        # sat and lght valleys
        C_mean, C_std = config.chroma_valley
        J_mean, J_std = config.lightness_valley
        for color in cs.colors:
            J, C, hue = color.JCh
            global_loss += ((J - J_mean) / J_std) ** 2 / 2 + ((C - C_mean) / C_std) ** 2 / 2
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
        if config.background is not None:
            for c in cs:
                global_loss += config.potential(c.delta_E(config.background))

        standard_pairwise_loss = pairwise_cost(cs)

        colorblind_pairwise_loss = 0
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
        x0=pack(init_colorset),
        method=config.minimization_algorithm,
    )
    if config.verbose:
        print(res)
    result = unpack(res.x)
    assert result is not None, "Minimization result is not a valid color"
    return result
