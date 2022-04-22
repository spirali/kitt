from typing import List, Tuple

import seaborn as sns

BGRUintColor = Tuple[int, int, int]

RGBFloatColor = Tuple[float, float, float]

"""
Palette that can be used to assign different colors.
You can index the palette with a number.

Returns RGB values in range [0.0, 1.0].
"""
DEFAULT_PALETTE: List[RGBFloatColor] = sns.color_palette("pastel", 100)
