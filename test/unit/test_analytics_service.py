from dataclasses import dataclass
from re import M
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytest
from app.api.analytics.service import (_make_chart_dicts,
                                       _make_chart_dicts_for_boxplot,
                                       get_modelling_results,
                                       get_population_analytics,
                                       get_prediction_for_pipeline,
                                       get_quality_analytics)


def _assert_arrays(given: List[Any], correct: List[Any], label: str = 'x'):
    assert len(given) == len(correct), (
        f'{label} length error: len({label})={len(given)} != len(correct_{label})={len(correct)}'
    )
    if given and type(given[0]) is list:
        for idx, (giveni, correcti) in enumerate(zip(given, correct)):
            assert giveni == pytest.approx(correcti), (
                f'Incorrect {label}{idx} mapping: {label}{idx}={giveni} != correct_{label}{idx}={correcti}'
            )
    else:
        assert given == pytest.approx(correct), (
            f'Incorrect {label} mapping: {label}={given} != correct_{label}={correct}'
        )


@dataclass
class ChartTestCase:
    x: List[int]
    ys: List[List[Union[int, float]]]
    x_title: str
    y_title: str
    names: List[str]
    y_bnd: Optional[Tuple[int, int]]


CHART_TEST_CASES = [
    ChartTestCase(
        x=[1, 2, 3],
        ys=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        x_title='x',
        y_title='y',
        names=['test', 'sample', 'cases'],
        y_bnd=None
    ),
    ChartTestCase(
        x=[4, 5, 6],
        ys=[
            [1.2, 2.3, 3.4],
            [4.5, 5.6, 6.7],
            [7.8, 8.9, 9.10]
        ],
        x_title='x',
        y_title='y',
        names=['test', 'sample', 'cases'],
        y_bnd=(1.14, 9.555)
    ),
    ChartTestCase(
        x=[10, 50, 100],
        ys=[
            [0.00199, 0.01999, 0.19999],
            [0.5555, 0, 0],
            [7, 8, 9]
        ],
        x_title='x',
        y_title='y',
        names=['test', 'sample', 'cases'],
        y_bnd=(0, 9.45)
    )
]


@pytest.mark.parametrize('case', CHART_TEST_CASES)
def test_make_chart_dicts_for_boxplot(case: ChartTestCase):
    result = _make_chart_dicts_for_boxplot(
        x=case.x,
        ys=case.ys,
        x_title=case.x_title,
        y_title=case.y_title
    )
    correct = [
        {
            'y': f'Gen {case.x[idx]}',
            'x': [round(_, 3) for _ in y],
            'type': 'box',
            'name': idx
        }
        for idx, y in enumerate(case.ys)
    ]
    _assert_arrays(result, correct, 'chart')


@pytest.mark.parametrize('case', CHART_TEST_CASES)
def test_make_chart_dicts(case: ChartTestCase):
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
        correct_series = [
            {
                'name': case.names[idx1],
                'data':
                    [round(_, 3) for _ in y]
                    if plot_type == 'line'
                    else [[case.x[idx2], round(_, 3)] for idx2, _ in enumerate(y)]
            }
            for idx1, y in enumerate(case.ys)
        ]
        if not case.y_bnd:
            min_y: float = min(min(y) for y in case.ys) * 0.95
            max_y: float = max(max(y) for y in case.ys) * 1.05
        else:
            min_y, max_y = case.y_bnd
        correct_options = {
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
    _assert_arrays(result_series, correct_series, 'series')
    _assert_arrays(result_options, correct_options, 'options')


@pytest.fixture
def plot_data_fixture(monkeypatch):
    @dataclass
    class MockPlotData:
        series: list
        options: list

        def __iter__(self):
            return iter((self.series, self.options))

    monkeypatch.setattr('app.api.analytics.service.PlotData', MockPlotData)


@dataclass
class MockIndividualGraph:
    depth: int


@dataclass
class MockIndividual:
    fitness: float
    graph: MockIndividualGraph = None


@dataclass
class MockOptHistory:
    individuals: list


@pytest.fixture
def composer_history_for_case_fixture(monkeypatch):
    individuals = [
        [
            MockIndividual(fitness, MockIndividualGraph(depth))
            for fitness, depth in ind_lst
        ]
        for ind_lst in [
            [(1.1111, 1), (2.2229999, 4)],
            [(3.3366, 9)],
            [(5.54321, 25), (-6.123456, 36), (10.54346, 100)]
        ]
    ]

    def mock_composer_history_for_case(case_id: str):
        return MockOptHistory(individuals)
    monkeypatch.setattr(
        'app.api.analytics.service.composer_history_for_case', mock_composer_history_for_case
    )


@pytest.fixture
def chart_dicts_fixture(monkeypatch):
    def mock_make_chart_dicts(x, ys, *args, **kwargs):
        return x, ys
    monkeypatch.setattr(
        'app.api.analytics.service._make_chart_dicts', mock_make_chart_dicts
    )

    def mock_make_chart_dicts_for_boxplot(x, ys, *args, **kwargs) -> Tuple[list, list]:
        return x, ys
    monkeypatch.setattr(
        'app.api.analytics.service._make_chart_dicts_for_boxplot', mock_make_chart_dicts_for_boxplot
    )


def test_get_quality_analytics(
    plot_data_fixture,
    composer_history_for_case_fixture,
    chart_dicts_fixture
):
    x, y = get_quality_analytics('_')
    y = y[0]
    assert x and y, 'MockPlotData is empty'
    assert len(x) == len(y), 'x and y should have the same shape'
    correct_x = [0, 1, 2]
    _assert_arrays(x, correct_x, 'x')
    correct_y = [1.111, 3.337, 6.123]
    _assert_arrays(y, correct_y, 'y')


@pytest.fixture
def box_plot_data_fixture(monkeypatch):
    @dataclass
    class MockBoxPlotData:
        series: Tuple[list, list]

        def __iter__(self):
            return iter(self.series)

    monkeypatch.setattr('app.api.analytics.service.BoxPlotData', MockBoxPlotData)


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
    composer_history_for_case_fixture,
    chart_dicts_fixture,
    box_plot_data_fixture
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
class MockMetaData:
    dataset_name: str = ''
    task_name: str = ''


@dataclass
class MockShowcaseItem:
    metadata: MockMetaData = MockMetaData()


@pytest.fixture
def showcase_item_fixture(monkeypatch):
    monkeypatch.setattr(
        'app.api.analytics.service.ShowcaseItem', MockShowcaseItem
    )


@dataclass
class MockInputData:
    pass


@dataclass
class MockOutputData:
    predict: np.ndarray = None


@pytest.fixture
def input_output_data_fixture(monkeypatch):
    monkeypatch.setattr(
        'app.api.analytics.service.InputData', MockInputData
    )
    monkeypatch.setattr(
        'app.api.analytics.service.OutputData', MockOutputData
    )


@dataclass
class MockPipeline:
    is_fitted: bool = False
    should_return_baseline: bool = False  # non-existing field

    def fit(self, *args, **kwargs):
        self.is_fitted = True

    def predict(self, *args, **kwargs):
        if not self.is_fitted:
            raise ValueError()
        return MockOutputData()


@pytest.fixture
def pipeline_fixture(monkeypatch):
    monkeypatch.setattr(
        'app.api.analytics.service.Pipeline', MockPipeline
    )


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
    showcase_item_fixture,
    input_output_data_fixture,
    chart_dicts_fixture,
    pipeline_fixture
):
    def mock_get_input_data(*args, **kwargs):
        if kwargs['dataset_name'] is None:
            return None
        return MockInputData()
    monkeypatch.setattr(
        'app.api.analytics.service.get_input_data', mock_get_input_data
    )
    if type(case.target) is str:
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
    chart_dicts_fixture,
    showcase_item_fixture
):
    def mock_get_prediction_for_pipeline(showcase: MockShowcaseItem, pipeline: MockPipeline, *args, **kwargs):
        if pipeline:
            if pipeline.should_return_baseline:
                if showcase.metadata.task_name == 'ts_forecasting':
                    return None, MockOutputData(np.array([[10, 11, 12, 13, 14, 15]]))
                return None, MockOutputData(np.array([[22], [33], [44], [55], [66], [77]]))
            else:
                if showcase.metadata.task_name == 'ts_forecasting':
                    return None, MockOutputData(np.array([[1, 2, 3, 4, 5, 6]]))
                return None, MockOutputData(np.array([[1], [2], [3], [4], [5], [6]]))
        return None, None
    monkeypatch.setattr(
        'app.api.analytics.service.get_prediction_for_pipeline', mock_get_prediction_for_pipeline
    )

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
