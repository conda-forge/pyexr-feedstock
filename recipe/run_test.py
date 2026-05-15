import tempfile
from pathlib import Path

import numpy as np
import pyexr


image = np.zeros((4, 5, 3), dtype=np.float32)
image[..., 0] = 0.25
image[..., 1] = np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32)
image[..., 2] = np.linspace(1.0, 0.0, image.shape[0], dtype=np.float32)[:, None]

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "roundtrip.exr"
    pyexr.write(str(path), image)
    loaded = pyexr.read(str(path))

assert loaded.shape == image.shape
assert loaded.dtype == np.float32
assert np.allclose(loaded, image)
