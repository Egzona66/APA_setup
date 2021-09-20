from myterial import (
    light_blue,
    light_blue_dark,
    orange,
    orange_dark,
    indigo,
    indigo_dark,
    salmon,
    salmon_dark,
    blue_grey,
    blue_grey_dark,
)
import numpy as np

sensors = ["fl", "fr", "hl", "hr"]

colors = dict(
    fr=light_blue, fl=orange, hr=indigo, hl=salmon, tot_weight=blue_grey, CoG=blue_grey,
)

dark_colors = dict(
    fr=light_blue_dark,
    fl=orange_dark,
    hr=indigo_dark,
    hl=salmon_dark,
    tot_weight=blue_grey_dark,
    CoG=blue_grey_dark,
)

# XY coordinates in CM of each sensor from the center of the array
sensors_vectors = dict(
    fr=np.array([0.75, 1.625]),
    fl=np.array([-0.75, 1.625]),
    hr=np.array([1, -1.625]),
    hl=np.array([-1, -1.625]),
)
