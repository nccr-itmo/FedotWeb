from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from app.api.analytics.service import (_make_chart_dicts,
                                       _make_chart_dicts_for_boxplot,
                                       get_modelling_results,
                                       get_population_analytics,
                                       get_prediction_for_pipeline,
                                       get_quality_analytics)
from pandas.testing import assert_series_equal

from .fixtures.analytics_service_fixtures import (
    box_plot_data_fixture, composer_history_for_case_fixture,
    input_output_data_fixture, make_chart_dicts_fixture, pipeline_fixture,
    pipeline_prediction_fixture, plot_data_fixture, showcase_item_fixture)
from .mocks.analytics_service_mocks import (MockInputData, MockMetaData,
                                            MockOutputData, MockPipeline,
                                            MockShowcaseItem)


def _assert_arrays(given: List[Any], correct: List[Any], label: str = 'x'):
    assert len(given) == len(correct), (
        f'{label} length error: len({label})={len(given)} != len(correct_{label})={len(correct)}'
    )
    if given and type(given[0]) is list:
        for idx, (giveni, correcti) in enumerate(zip(given, correct)):
            assert giveni == pytest.approx(correcti), (
                f'Incorrect {label}{idx} mapping: {label}{idx}={giveni} != correct_{label}{idx}={correcti}'
            )
    elif given and type(given[0]) is dict:
        assert_series_equal(pd.Series(given), pd.Series(correct))
    else:
        assert given == pytest.approx(correct), (
            f'Incorrect {label} mapping: {label}={given} != correct_{label}={correct}'
        )


@dataclass
class BoxplotChartTestCase:
    x: List[int]
    ys: List[List[Union[int, float]]]
    x_title: str
    y_title: str
    correct_output: List[Dict]


BOXPLOT_CHART_TEST_CASES = [
    BoxplotChartTestCase(
        x=[1, 2, 3],
        ys=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        x_title='x',
        y_title='y',
        correct_output=[
            {
                'x': [1, 2, 3],
                'y': 'Gen 1',
                'type': 'box',
                'name': 0
            },
            {
                'x': [4, 5, 6],
                'y': 'Gen 2',
                'type': 'box',
                'name': 1
            },
            {
                'x': [7, 8, 9],
                'y': 'Gen 3',
                'type': 'box',
                'name': 2
            }
        ],
    ),
    BoxplotChartTestCase(
        x=[4, 5, 6],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.10]
        ],
        x_title='x',
        y_title='y',
        correct_output=[
            {
                'x': [1.2, 2.3, 3.4],
                'y': 'Gen 4',
                'type': 'box',
                'name': 0
            },
            {
                'x': [4.5, 5.6, 6.7],
                'y': 'Gen 5',
                'type': 'box',
                'name': 1
            },
            {
                'x': [7.8, 8.9, 9.10],
                'y': 'Gen 6',
                'type': 'box',
                'name': 2
            }
        ]
    ),
    BoxplotChartTestCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ],
        x_title='x',
        y_title='y',
        correct_output=[
            {
                'x': [0.002, 0.02, 0.2],
                'y': 'Gen 10',
                'type': 'box',
                'name': 0
            },
            {
                'x': [0.555, 0, 0],
                'y': 'Gen 50',
                'type': 'box',
                'name': 1
            },
            {
                'x': [7, 8, 9],
                'y': 'Gen 100',
                'type': 'box',
                'name': 2
            }
        ],
    )
]


@pytest.mark.parametrize('case', BOXPLOT_CHART_TEST_CASES)
def test_make_chart_dicts_for_boxplot(case: BoxplotChartTestCase):
    result = _make_chart_dicts_for_boxplot(
        x=case.x,
        ys=case.ys,
        x_title=case.x_title,
        y_title=case.y_title
    )
    _assert_arrays(result, case.correct_output, 'chart')


@dataclass
class ChartTestCase:
    x: List[int]
    ys: List[List[Union[int, float]]]
    x_title: str
    y_title: str
    names: List[str]
    plot_type: str
    y_bnd: Optional[Tuple[int, int]]
    correct_series: List[Dict]
    correct_options: Dict


CHART_TEST_CASES = [
    ChartTestCase(
        x=[1, 2, 3],
        ys=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        x_title='x1',
        y_title='y1',
        names=['test1', 'sample1', 'cases1'],
        plot_type='other',
        y_bnd=None,
        correct_series=[
            {
                'name': 'test1',
                'data': [
                    [1, 1],
                    [2, 2],
                    [3, 3]
                ]
            },
            {
                'name': 'sample1',
                'data': [
                    [1, 4],
                    [2, 5],
                    [3, 6]
                ]
            },
            {
                'name': 'cases1',
                'data': [
                    [1, 7],
                    [2, 8],
                    [3, 9]
                ]
            }
        ],
        correct_options={
            'chart': {
                'type': 'other'
            },
            'xaxis': {
                'categories': [1, 2, 3],
                'title': {
                    'text': 'x1'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y1'
                },
                'min': 0.95,
                'max': 9.45
            }
        }
    ),
    ChartTestCase(
        x=[1, 2, 3, 4],
        ys=[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        x_title='x2',
        y_title='y2',
        names=['test2', 'sample2', 'cases2'],
        plot_type='line',
        y_bnd=None,
        correct_series=[
            {
                'name': 'test2',
                'data': [1, 2, 3, 4]
            },
            {
                'name': 'sample2',
                'data': [5, 6, 7, 8]
            },
            {
                'name': 'cases2',
                'data': [9, 10, 11, 12]
            }
        ],
        correct_options={
            'chart': {
                'type': 'line'
            },
            'xaxis': {
                'categories': [1, 2, 3, 4],
                'title': {
                    'text': 'x2'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y2'
                },
                'min': 0.95,
                'max': 12.6
            }
        }
    ),

    ChartTestCase(
        x=[0, 1, 2],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.1]
        ],
        x_title='x3',
        y_title='y3',
        names=['test3', 'sample3', 'cases3'],
        plot_type='other',
        y_bnd=(1.14, 9.555),
        correct_series=[
            {
                'name': 'test3',
                'data': [
                    [0, 1.2],
                    [1, 2.3],
                    [2, 3.4]
                ]
            },
            {
                'name': 'sample3',
                'data': [
                    [0, 4.5],
                    [1, 5.6],
                    [2, 6.7]
                ]
            },
            {
                'name': 'cases3',
                'data': [
                    [0, 7.8],
                    [1, 8.9],
                    [2, 9.1]
                ]
            }
        ],
        correct_options={
            'chart': {
                'type': 'other'
            },
            'xaxis': {
                'categories': [0, 1, 2],
                'title': {
                    'text': 'x3'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y3'
                },
                'min': 1.14,
                'max': 9.555
            }
        }
    ),
    ChartTestCase(
        x=[0, 1, 2],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.1]
        ],
        x_title='x4',
        y_title='y4',
        names=['test4', 'sample4', 'cases4'],
        plot_type='line',
        y_bnd=(1.14, 9.555),
        correct_series=[
            {
                'name': 'test4',
                'data': [1.2, 2.3, 3.4]
            },
            {
                'name': 'sample4',
                'data': [4.5, 5.6, 6.7]
            },
            {
                'name': 'cases4',
                'data': [7.8, 8.9, 9.1]
            }
        ],
        correct_options={
            'chart': {
                'type': 'line'
            },
            'xaxis': {
                'categories': [0, 1, 2],
                'title': {
                    'text': 'x4'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y4'
                },
                'min': 1.14,
                'max': 9.555
            }
        }
    ),

    ChartTestCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ],
        x_title='x5',
        y_title='y5',
        names=['test5', 'sample5', 'cases5'],
        plot_type='other',
        y_bnd=(0, 9.45),
        correct_series=[
            {
                'name': 'test5',
                'data': [
                    [10, 0.002],
                    [50, 0.02],
                    [100, 0.2]
                ]
            },
            {
                'name': 'sample5',
                'data': [
                    [10, 0.555],
                    [50, 0],
                    [100, 0]
                ]
            },
            {
                'name': 'cases5',
                'data': [
                    [10, 7],
                    [50, 8],
                    [100, 9]
                ]
            }
        ],
        correct_options={
            'chart': {
                'type': 'other'
            },
            'xaxis': {
                'categories': [10, 50, 100],
                'title': {
                    'text': 'x5'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y5'
                },
                'min': 0,
                'max': 9.45
            }
        }
    ),
    ChartTestCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ],
        x_title='x6',
        y_title='y6',
        names=['test6', 'sample6', 'cases6'],
        plot_type='line',
        y_bnd=None,
        correct_series=[
            {
                'name': 'test6',
                'data': [0.002, 0.02, 0.2]
            },
            {
                'name': 'sample6',
                'data': [0.555, 0, 0]
            },
            {
                'name': 'cases6',
                'data': [7, 8, 9]
            }
        ],
        correct_options={
            'chart': {
                'type': 'line'
            },
            'xaxis': {
                'categories': [10, 50, 100],
                'title': {
                    'text': 'x6'
                }
            },
            'yaxis': {
                'title': {
                    'text': 'y6'
                },
                'min': 0,
                'max': 9.45
            }
        }
    )
]


@pytest.mark.parametrize('case', CHART_TEST_CASES)
def test_make_chart_dicts(case: ChartTestCase):
    result_series, result_options = _make_chart_dicts(
        x=case.x,
        ys=case.ys,
        names=case.names,
        x_title=case.x_title,
        y_title=case.y_title,
        plot_type=case.plot_type,
        y_bnd=case.y_bnd
    )
    _assert_arrays(result_series, case.correct_series, 'series')
    _assert_arrays([result_options], [case.correct_options], 'options')


def test_get_quality_analytics(
    plot_data_fixture,
    composer_history_for_case_fixture,
    make_chart_dicts_fixture
):
    x, y = get_quality_analytics('_')
    y = y[0]
    assert x and y, 'MockPlotData is empty'
    assert len(x) == len(y), 'x and y should have the same shape'
    correct_x = [0, 1, 2]
    _assert_arrays(x, correct_x, 'x')
    correct_y = [1.111, 3.337, 6.123]
    _assert_arrays(y, correct_y, 'y')


@dataclass
class PopulationAnalyticsTestCase:
    analytic_type: str
    correct_x: List[int]
    correct_y: List[Any]


POPULATION_ANALYTICS_TEST_CASES = [
    PopulationAnalyticsTestCase(
        analytic_type='pheno',
        correct_x=[0, 1, 2],
        correct_y=[[1.1111, 2.2229999], [3.3366], [5.54321, 6.123456, 10.54346]]
    ),
    PopulationAnalyticsTestCase(
        analytic_type='geno',
        correct_x=[0, 1, 2],
        correct_y=[[1, 4], [9], [25, 36, 100]]
    ),
    PopulationAnalyticsTestCase(
        analytic_type='exception',
        correct_x=[],
        correct_y=[]
    )
]


@pytest.mark.parametrize('case', POPULATION_ANALYTICS_TEST_CASES)
def test_get_population_analytics(
    case: PopulationAnalyticsTestCase,
    monkeypatch,
    box_plot_data_fixture,
    make_chart_dicts_fixture,
    composer_history_for_case_fixture,
):
    if case.analytic_type == 'exception' and not case.correct_x and not case.correct_y:
        with pytest.raises(ValueError):
            get_population_analytics('', case.analytic_type)
    else:
        x, y = get_population_analytics('', case.analytic_type)
        assert x and y, 'MockBoxPlotData is empty'
        assert len(x) == len(y), 'x and y should have the same shape'
        _assert_arrays(x, case.correct_x, 'x')
        _assert_arrays(y, case.correct_y, 'y')


@dataclass
class PipelinePredictionTestCase:
    showcase: MockShowcaseItem
    pipeline: Optional[MockPipeline]
    target: Union[str, Callable[[Any], Any]]


PIPELINE_PREDICTION_TEST_CASES = [
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=None,
        target=lambda test_data, prediction: test_data is None and prediction is None
    ),
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=MockPipeline(),
        target='exception'
    ),
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=MockPipeline(is_fitted=True),
        target=lambda test_data, prediction: test_data is None and prediction is None
    ),
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(),
        pipeline=None,
        target=lambda test_data, prediction: type(test_data) is MockInputData and prediction is None
    ),
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(),
        pipeline=MockPipeline(),
        target=lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
    ),
    PipelinePredictionTestCase(
        showcase=MockShowcaseItem(),
        pipeline=MockPipeline(is_fitted=True),
        target=lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
    ),
]


@pytest.mark.parametrize('case', PIPELINE_PREDICTION_TEST_CASES)
def test_get_prediction_for_pipeline(
    case: PipelinePredictionTestCase,
    monkeypatch,
    pipeline_fixture,
    showcase_item_fixture,
    make_chart_dicts_fixture,
    input_output_data_fixture,
):
    if type(case.target) is str and case.target == 'exception':
        with pytest.raises(ValueError):
            get_prediction_for_pipeline(case.showcase, case.pipeline)
    else:
        case.target(*get_prediction_for_pipeline(case.showcase, case.pipeline))


@dataclass
class ModelResultsTestCase:
    task_name: str
    pipeline: Optional[MockPipeline]
    baseline_pipeline: Optional[MockPipeline]
    correct_x: List[int]
    correct_y: List[Any]


MODEL_RESULTS_TEST_CASES = [
    ModelResultsTestCase(
        task_name='classification',
        pipeline=None,
        baseline_pipeline=None,
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='classification',
        pipeline=None,
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='classification',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        correct_x=[0, 1, 2, 3, 4, 5],
        correct_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModelResultsTestCase(
        task_name='classification',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[0, 1, 2, 3, 4, 5],
        correct_y=[[1, 2, 3, 4, 5, 6], [22, 33, 44, 55, 66, 77]]
    ),

    ModelResultsTestCase(
        task_name='regression',
        pipeline=None,
        baseline_pipeline=None,
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='regression',
        pipeline=None,
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='regression',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        correct_x=[0, 1, 2, 3, 4, 5],
        correct_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModelResultsTestCase(
        task_name='regression',
        pipeline=MockPipeline,
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[0, 1, 2, 3, 4, 5],
        correct_y=[[1, 2, 3, 4, 5, 6], [22, 33, 44, 55, 66, 77]]
    ),

    ModelResultsTestCase(
        task_name='ts_forecasting',
        pipeline=None,
        baseline_pipeline=None,
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='ts_forecasting',
        pipeline=None,
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='ts_forecasting',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        correct_x=[0],
        correct_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModelResultsTestCase(
        task_name='ts_forecasting',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[0],
        correct_y=[[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]
    ),

    ModelResultsTestCase(
        task_name='exception',
        pipeline=None,
        baseline_pipeline=None,
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='exception',
        pipeline=None,
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='exception',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        correct_x=[],
        correct_y=[]
    ),
    ModelResultsTestCase(
        task_name='exception',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(should_return_baseline=True),
        correct_x=[],
        correct_y=[]
    ),
]


@pytest.mark.parametrize('case', MODEL_RESULTS_TEST_CASES)
def test_get_modelling_results(
    case: ModelResultsTestCase,
    monkeypatch,
    plot_data_fixture,
    showcase_item_fixture,
    make_chart_dicts_fixture,
    pipeline_prediction_fixture
):
    showcase = MockShowcaseItem(MockMetaData('', case.task_name))
    if case.task_name == 'exception':
        with pytest.raises(NotImplementedError):
            get_modelling_results(showcase, case.pipeline, case.baseline_pipeline)
    elif case.pipeline is None:
        with pytest.raises(AttributeError):
            get_modelling_results(showcase, case.pipeline, case.baseline_pipeline)
    else:
        x, y = get_modelling_results(showcase, case.pipeline, case.baseline_pipeline)
        _assert_arrays(x, case.correct_x, 'x')
        _assert_arrays(y, case.correct_y, 'y')
