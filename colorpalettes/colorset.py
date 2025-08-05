import functools
import itertools
from dataclasses import dataclass
from typing import Iterator, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from colorpalettes.color import Color, ColorDeficiencyType


@dataclass(frozen=True)
class Colorset:
    colors: tuple[Color, ...]

    def __iter__(self) -> Iterator[Color]:
        return iter(self.colors)

    @staticmethod
    def from_cmap(name: str) -> "Colorset":
        cmap = matplotlib.colormaps[name]
        if isinstance(cmap, ListedColormap):
            return Colorset([Color(rgb=rgb) for rgb in cmap.colors])  # type: ignore
        else:
            raise ValueError(f"Not a listed colormap ({name}): {cmap}")

    @property
    def n(self) -> int:
        return len(self.colors)

    @functools.cached_property
    def distance_mat(self) -> np.ndarray:
        distances = np.zeros(shape=(self.n, self.n))

        for i, c1 in enumerate(self.colors):
            for j, c2 in enumerate(self.colors):
                if j >= i:
                    continue
                d = c1.delta_E(c2)
                distances[i, j] = d
                distances[j, i] = d
        return distances

    def print_distance_stats(self) -> None:
        distances = np.triu(self.distance_mat).flatten()
        distances = distances[distances > 0]
        print(
            f"D = {np.mean(distances):.2f} +/- {np.std(distances):.2f}; "
            + f"median = {np.median(distances):.2f}; "
            + f"worst = {np.min(distances):.2f}"
        )

    def color_deficient(self, type: ColorDeficiencyType) -> "Colorset":
        return Colorset(tuple(c.color_deficient(type=type) for c in self.colors))

    def sorted_by_lightness(self, reverse: bool = False) -> "Colorset":
        return Colorset(tuple(sorted(self.colors, key=lambda c: c.JCh[0], reverse=reverse)))

    def sorted_by_distance(self, closest: bool = True) -> "Colorset":
        colors = list(self.colors)
        sorted = [colors.pop(0)]
        while colors:
            ref = sorted[-1]
            delta_from_ref = [c.delta_E(ref) for c in colors]
            idx = np.argmin(delta_from_ref) if closest else np.argmax(delta_from_ref)
            sorted.append(colors.pop(idx))
        return Colorset(tuple(sorted))

    def plot_distances(self) -> Figure:
        fig, ax = plt.subplots()
        mat = self.distance_mat
        for i in range(self.n):
            mat[i, i] = np.nan
        mesh = ax.pcolormesh(mat, cmap="inferno", vmin=0.0, vmax=100.0)
        fig.colorbar(mesh)
        for i, color in enumerate(self.colors):
            ax.axvspan(i, i + 1, 0.0, 0.02, color=color.rgb)
            ax.axhspan(i, i + 1, 0.0, 0.02, color=color.rgb)

        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def plot_colors(self) -> Figure:
        fig = self.plot_colorblindness_check([])
        fig.axes[0].set_yticks([])
        return fig

    def plot_colorblindness_check(self, cb_types: Sequence[ColorDeficiencyType] = tuple(ColorDeficiencyType)) -> Figure:
        nrows = len(cb_types) + 1
        ncols = len(self.colors)

        fig, ax = plt.subplots(figsize=(8, nrows))

        for row_idx, colorset in enumerate(
            itertools.chain([self], [self.color_deficient(cb_type) for cb_type in cb_types])
        ):
            for i, color in enumerate(colorset.colors):
                ax.add_patch(
                    Rectangle(
                        xy=(i, nrows - row_idx - 1),
                        width=1,
                        height=1,
                        color=color.rgb,
                    )
                )
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks([])
        ax.set_yticks(
            [i + 0.5 for i in range(nrows)],
            reversed(["Unmodified"] + list(cb_types)),
            rotation=45,
            fontsize="small",
        )
        return fig

    def plot_all(self) -> None:
        self.plot_colors()
        self.plot_distances()
        self.plot_colorblindness_check()
