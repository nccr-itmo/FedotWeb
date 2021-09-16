from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
    assert result == etalon, f"Chart dicts aren't equal: {result} != {etalon}"


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
    assert result_series == etalon_series, f"Series aren't equal: {result_series} != {etalon_series}"
    assert result_options == etalon_options, f"Options aren't equal: {result_options} != {etalon_options}"


def test_get_quality_analytics(monkeypatch):
    @dataclass
    class MockIndividualFitness:
        fitness: int

    individuals = [
        [
            MockIndividualFitness(fitness)
            for fitness in fitness_lst
        ]
        for fitness_lst in [[1.1111, 2.2229999], [3.3366], [5.54321, 6.123456, 10.54346]]
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
        return x, ys[0]
    monkeypatch.setattr(
        "app.api.analytics.service._make_chart_dicts", mock_make_chart_dicts
    )

    @dataclass
    class MockPlotData:
        series: list
        options: list

        def __iter__(self):
            return iter((self.series, self.options))

    monkeypatch.setattr("app.api.analytics.service.PlotData", MockPlotData)

    x, y = get_quality_analytics("_")
    assert x and y, "MockPlotData is empty"
    assert len(x) == len(y), "x and y should have the same shape"
    etalon_x = [0, 1, 2]
    assert x == etalon_x, f"Incorrect x mapping: {x} != {etalon_x}"
    etalon_y = [1.111, 3.337, 5.543]
    assert y == etalon_y, f"Incorrect y mapping: {y} != {etalon_y}"


def test_test_prediction_for_pipeline(monkeypatch):
    @dataclass
    class MockInputData:
        val: str = ""
    monkeypatch.setattr(
        "app.api.analytics.service.InputData", MockInputData
    )

    def mock_get_input_data(*args, **kwargs):
        if kwargs["dataset_name"] is None:
            return None
        return MockInputData()
    monkeypatch.setattr(
        "app.api.analytics.service.get_input_data", mock_get_input_data
    )

    @dataclass
    class MockMetaData:
        dataset_name: str = ""

    @dataclass
    class MockShowcaseItem:
        metadata: MockMetaData = MockMetaData()
    monkeypatch.setattr(
        "app.api.analytics.service.ShowcaseItem", MockShowcaseItem
    )

    @dataclass
    class MockOutputData:
        pass

    @dataclass
    class MockPipeline:
        is_fitted: bool = False

        def fit(self, *args, **kwargs):
            self.is_fitted = True

        def predict(self, *args, **kwargs):
            if not self.is_fitted:
                raise ValueError()
            return MockOutputData()
    monkeypatch.setattr(
        "app.api.analytics.service.Pipeline", MockPipeline
    )
    test_inputs = [  # tests all paths covered
        (
            MockShowcaseItem(MockMetaData(None)), None,
            lambda test_data, prediction: test_data is None and prediction is None
        ),
        (
            MockShowcaseItem(MockMetaData(None)), MockPipeline(),
            "exception"
        ),
        (
            MockShowcaseItem(MockMetaData(None)), MockPipeline(is_fitted=True),
            lambda test_data, prediction: test_data is None and type(prediction) is MockOutputData
        ),
        (
            MockShowcaseItem(), None,
            lambda test_data, prediction: type(test_data) is MockInputData and prediction is None
        ),
        (
            MockShowcaseItem(), MockPipeline(),
            lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
        ),
        (
            MockShowcaseItem(), MockPipeline(is_fitted=True),
            lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
        )
    ]
    for case, pipeline, target in test_inputs:
        if type(target) is str:
            with pytest.raises(ValueError):
                _test_prediction_for_pipeline(case, pipeline)
        else:
            target(*_test_prediction_for_pipeline(case, pipeline))
