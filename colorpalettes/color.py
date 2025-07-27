import enum
from dataclasses import dataclass

import distinctipy
import numpy as np
from basic_colormath import get_delta_e_lab, hsl_to_rgb, rgb_to_hsl, rgb_to_lab


class ColorblindnessType(enum.StrEnum):
    Deuteranomaly = "Deuteranomaly"
    Protanopia = "Protanopia"
    Protanomaly = "Protanomaly"
    Deuteranopia = "Deuteranopia"


@dataclass
class Color:
    rgb: tuple[float, float, float]  # range: [0, 255.0], [0, 255.0], [0, 255.0]

    def __post_init__(self) -> None:
        self.rgb = tuple(np.clip(self.rgb, 0.0, 255.0))
        self.rgb_normalized = tuple(c / 255.0 for c in self.rgb)
        # range: [0, 100], [-128, 127], [-128, 127]
        self.lab = rgb_to_lab(self.rgb)
        # range: [0, 360), [0, 100], [0, 100]
        self.hsl = rgb_to_hsl(self.rgb)

    @staticmethod
    def from_rgb_normalized(rgb_0_1: tuple[float, float, float]) -> "Color":
        return Color(tuple(255.0 * c for c in rgb_0_1))  # type: ignore

    def not_extreme(self, eps: float = 0.1) -> "Color":
        return Color(tuple(np.clip(self.rgb, eps, 255.0 - eps)))

    @staticmethod
    def from_hsl(hsl: tuple[float, float, float]) -> "Color":
        hue, sat, lght = hsl
        hue = hue % 360.0
        sat = np.clip(sat, 0.0, 100.0)
        lght = np.clip(lght, 0.0, 100.0)
        return Color(hsl_to_rgb((hue, sat, lght)))

    @staticmethod
    def white() -> "Color":
        return Color.from_rgb_normalized((1.0, 1.0, 1.0))

    @staticmethod
    def black() -> "Color":
        return Color.from_rgb_normalized((0.0, 0.0, 0.0))

    @staticmethod
    def red() -> "Color":
        return Color.from_rgb_normalized((1.0, 0.0, 0.0))

    @staticmethod
    def green() -> "Color":
        return Color.from_rgb_normalized((0.0, 1.0, 0.0))

    @staticmethod
    def blue() -> "Color":
        return Color.from_rgb_normalized((0.0, 0.0, 1.0))

    def lab_phase(self) -> float:
        phi = np.atan(self.lab[2] / self.lab[1])
        if self.lab[1] < 0:
            phi += np.pi
        elif phi < 0:
            phi += 2 * np.pi
        return phi

    def lab_magnitude(self) -> float:
        return np.sqrt(self.lab[2] ** 2 + self.lab[1] ** 2)

    def delta_E(self, other: "Color") -> float:
        return get_delta_e_lab(self.lab, other.lab)

    def colorblinded(self, type: ColorblindnessType) -> "Color":
        return Color.from_rgb_normalized(
            distinctipy.colorblind.colorblind_filter(
                self.rgb_normalized,
                colorblind_type=str(type).capitalize(),
            )
        )
