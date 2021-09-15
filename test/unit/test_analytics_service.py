from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import app.api.analytics.models
import app.api.analytics.service
import app.api.composer.service
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
    x_title: str
    y_title: str
    names: List[str]
    y_bnd: Optional[Tuple[int, int]]


InputCases = [
    InputCase(
        x=[1, 2, 3],
        ys=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        x_title="x",
        y_title="y",
        names=["test", "sample", "cases"],
        y_bnd=None
    ),
    InputCase(
        x=[4, 5, 6],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.10]
        ],
        x_title="x",
        y_title="y",
        names=["test", "sample", "cases"],
        y_bnd=(1.14, 9.555)
    ),
    InputCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ],
        x_title="x",
        y_title="y",
        names=["test", "sample", "cases"],
        y_bnd=(0, 9.45)
    )
]


@pytest.mark.parametrize("case", InputCases)
def test_make_chart_dicts_for_boxplot(case: InputCase):
    result = _make_chart_dicts_for_boxplot(
        x=case.x,
        ys=case.ys,
        x_title=case.x_title,
        y_title=case.y_title
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
    assert result == etalon


@pytest.mark.parametrize("case", InputCases)
def test_make_chart_dicts(case: InputCase):
    plot_types = ['line', 'other']
    for plot_type in plot_types:
        result_series, result_options = _make_chart_dicts(
            x=case.x,
            ys=case.ys,
            names=case.names,
            x_title=case.x_title,
            y_title=case.y_title,
            plot_type=plot_type,
            y_bnd=case.y_bnd
        )
        etalon_series = [
            {
                'name': case.names[idx1],
                'data':
                    [round(_, 3) for _ in y]
                    if plot_type == "line"
                    else [[case.x[idx2], round(_, 3)] for idx2, _ in enumerate(y)]
            }
            for idx1, y in enumerate(case.ys)
        ]
        if not case.y_bnd:
            min_y: float = min(min(y) for y in case.ys) * 0.95
            max_y: float = max(max(y) for y in case.ys) * 1.05
        else:
            min_y, max_y = case.y_bnd
        etalon_options = {
            'chart': {
                'type': plot_type
            },
            'xaxis': {
                'categories': case.x,
                'title': {
                    'text': case.x_title
                }
            },
            'yaxis': {
                'title': {
                    'text': case.y_title
                },
                'min': min_y,
                'max': max_y
            }
        }
    assert result_series == etalon_series, "Series aren't equal"
    assert result_options == etalon_options, "Options aren't equal"


def test_get_quality_analytics(monkeypatch):
    individuals = [
        1.1111, 2.2229999, 3.3366, 5.54321, 6.123456
    ]

    @dataclass
    class MockOptHistory:
        individuals: list

    def mock_composer_history_for_case(case_id: str):
        history = MockOptHistory(individuals)
        return history
    monkeypatch.setattr(
        "app.api.analytics.service.composer_history_for_case", mock_composer_history_for_case
    )

    def mock_make_chart_dicts(x, ys, names, x_title, y_title, plot_type):
        return x, ys
    monkeypatch.setattr(
        "app.api.analytics.service._make_chart_dicts", mock_make_chart_dicts
    )

    @dataclass
    class MockPlotData:
        x: list
        y: list
    monkeypatch.setattr("app.api.analytics.service.PlotData", MockPlotData)

    result: MockPlotData = get_quality_analytics("_")
    assert result.x and result.y, "MockPlotData is empty"
    assert result.x == [0, 1, 2, 3, 4], "Incorrect x mapping"
    assert result.y == [1.111, 2.222, 3.337, 5.543, 6.123], "Incorrect y mapping"
