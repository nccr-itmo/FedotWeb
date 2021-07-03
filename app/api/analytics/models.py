from dataclasses import dataclass
from typing import List


@dataclass
class PlotData:
    series: List[dict]
    options: dict
