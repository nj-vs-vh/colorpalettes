import enum
import functools
from dataclasses import dataclass

from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
from colorspacious import cspace_convert, deltaE


class ColorDeficiencyType(enum.StrEnum):
    Deuteranomaly = "Deuteranomaly"
    Protanomaly = "Protanomaly"


@dataclass(frozen=True)
class Color:
    rgb: tuple[float, float, float]  # RGB1 representation
    validated: bool = True

    JCh_precomputed: tuple[float, float, float] | None = None
    CAM02UCS_precomputed: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.validated and (not all(0 <= c <= 1 for c in self.rgb) or any(np.isnan(c) for c in self.rgb)):
            raise ValueError(f"Invalid RGB coordinate: {self.rgb}")

    @functools.cached_property
    def JCh(self) -> tuple[float, float, float]:
        if self.JCh_precomputed is not None:
            return self.JCh_precomputed
        # range: [0, ~100], [0, ~100], [0, 360]
        return cspace_convert(self.rgb, "sRGB1", "JCh")

    @staticmethod
    def from_JCh(JCh: tuple[float, float, float], ensure_valid: bool = True) -> "Color":
        rgb = cspace_convert(JCh, "JCh", "sRGB1")
        if ensure_valid:
            rgb = tuple(np.clip(rgb, 0, 1))
        return Color(rgb, JCh_precomputed=JCh)

    @functools.cached_property
    def CAM02UCS(self) -> tuple[float, float, float]:
        return cspace_convert(self.JCh, "JCh", "CAM02-UCS")

    @staticmethod
    def from_CAM02UCS(CAM02: tuple[float, float, float], ensure_valid: bool = True) -> "Color":
        rgb = cspace_convert(CAM02, "CAM02-UCS", "sRGB1")
        if ensure_valid:
            rgb = tuple(np.clip(rgb, 0, 1))
        return Color(rgb, CAM02UCS_precomputed=CAM02)

    @functools.cached_property
    def hsv(self) -> tuple[float, float, float]:
        return rgb_to_hsv(self.rgb)  # type: ignore

    @staticmethod
    def from_hsv(hsv: tuple[float, float, float], ensure_valid: bool = True) -> "Color":
        rgb: tuple[float, float, float] = tuple(hsv_to_rgb(hsv))
        if ensure_valid:
            rgb = tuple(np.clip(rgb, 0, 1))
        return Color(rgb)

    @staticmethod
    def checked(rgb: tuple[float, float, float]) -> "Color":
        return Color(tuple(np.clip(rgb, 0.0, 1.0)))

    @staticmethod
    def white() -> "Color":
        return Color((1.0, 1.0, 1.0))

    @staticmethod
    def black() -> "Color":
        return Color((0.0, 0.0, 0.0))

    @staticmethod
    def red() -> "Color":
        return Color((1.0, 0.0, 0.0))

    @staticmethod
    def green() -> "Color":
        return Color((0.0, 1.0, 0.0))

    @staticmethod
    def blue() -> "Color":
        return Color((0.0, 0.0, 1.0))

    def delta_E(self, other: "Color") -> float:
        return deltaE(self.JCh, other.JCh, input_space="JCh")

    def color_deficient(self, type: ColorDeficiencyType, severity: float = 80) -> "Color":
        return Color.checked(
            rgb=cspace_convert(
                self.rgb,
                start={"name": "sRGB1+CVD", "cvd_type": type.lower(), "severity": severity},
                end="sRGB1",
            )
        )
