from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import pytest
from app.api.analytics.service import (_make_chart_dicts,
                                       _make_chart_dicts_for_boxplot,
                                       _test_prediction_for_pipeline,
                                       get_modelling_results,
                                       get_population_analytics,
                                       get_quality_analytics)


@dataclass
class InputCase:
    x: List[int]
    ys: List[List[Union[int, float]]]

InputCases = [
    InputCase(
        x=[1, 2, 3],
        ys=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    ),
    InputCase(
        x=[4, 5, 6],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.10]
        ]
    ),
    InputCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ]
    )
]

@pytest.mark.parametrize("case", InputCases)
def test_make_chart_dicts_for_boxplot(case: InputCase):
    result = _make_chart_dicts_for_boxplot(
        x=case.x,
        ys=case.ys,
        x_title="x",
        y_title="y"
    )
    etalon = [
        {
            'y': f'Gen {case.x[idx]}',
            'x': [round(_, 3) for _ in y],
            'type': 'box',
            'name': idx
        }
        for idx, y in enumerate(case.ys)
    ]
    assert sorted(result, key=lambda dct: dct.keys()) == sorted(etalon, key=lambda dct: dct.keys())
