import numpy as np
from PIL.Image import Image
from sklearn.cluster import MiniBatchKMeans

from colorpalettes.color import Color
from colorpalettes.colorset import Colorset


def image_colors(image: Image) -> np.ndarray:
    imdata = np.copy(np.asarray(image))
    if image.mode == "RGBA":
        np.delete(imdata, 3, 2)  # dropping the alpha channel
    elif image.mode != "RGB":
        raise ValueError("Only RGB and RGBA images are supported!")
    imdata = imdata.astype(float)
    return imdata.reshape(-1, 3)


def k_means_palette(image: Image, n_colors: int, random_seed: int | None = None) -> Colorset:
    colors_arr = image_colors(image)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=128, random_state=random_seed)
    kmeans.fit(colors_arr)
    palette = kmeans.cluster_centers_
    # darkness_idx = np.flip(np.argsort(np.sum(palette**2, axis=1)))
    return Colorset(tuple(Color(c) for c in palette))
