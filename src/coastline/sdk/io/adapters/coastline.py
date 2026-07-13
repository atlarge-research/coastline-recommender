"""The identity adapter: input is already (near-)canonical.

``to_canonical`` only normalizes accepted alias spellings (``model`` → ``llm_model``,
``gpu`` → ``gpu_model``, …) to their canonical column names; ``from_canonical`` is a
pass-through because the recommendation result is already in the canonical output schema.
This is the default adapter for the native Coastline batch CSV.
"""

from __future__ import annotations

import pandas as pd

from coastline.sdk.io import schema
from coastline.sdk.io.adapters.base import register


class CoastlineAdapter:
    name = "coastline"

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename any accepted alias columns to their canonical spelling; leave the rest."""
        renames = schema.canonicalize_columns(list(df.columns))
        # Don't collide with a column that is already canonically named.
        renames = {src: dst for src, dst in renames.items() if src == dst or dst not in df.columns}
        return df.rename(columns=renames)

    def from_canonical(self, recommended: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """The recommendation frame is already canonical output — pass it through."""
        return recommended


register(CoastlineAdapter())
