from PIL import Image
import numpy as np

from sklearn.cluster import MiniBatchKMeans

from typing import Any
from nptyping import NDArray


def image_to_colorset(image: Image) -> NDArray[(Any, 3), float]:
    imdata = np.copy(np.asarray(image))
    if image.mode == 'RGBA':
        np.delete(imdata, 3, 2)  # dropping the alpha channel
    elif image.mode != 'RGB':
        raise ValueError("Only RGB and RGBA images are supported!")
    imdata = imdata.astype(float)
    imdata /= 255  # PIL's RGB and RGBA modes ensure 8-bit depth
    return imdata.reshape(-1, 3)


def extract_palette(image: Image, n_colors: int = 10) -> NDArray[(Any, 3), float]:
    colorset = image_to_colorset(image)
    kmeans = MiniBatchKMeans(n_clusters=n_colors)
    kmeans.fit(colorset)
    palette = kmeans.cluster_centers_
    darkness_idx = np.flip(np.argsort(np.sum(palette ** 2, axis=1)))
    return palette[darkness_idx, :]
