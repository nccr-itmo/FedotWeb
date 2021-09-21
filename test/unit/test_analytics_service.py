from dataclasses import dataclass
from re import M
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from _pytest.monkeypatch import monkeypatch
from app.api.analytics.service import (_make_chart_dicts,
                                       _make_chart_dicts_for_boxplot,
                                       _test_prediction_for_pipeline,
                                       get_modelling_results,
                                       get_population_analytics,
                                       get_quality_analytics)
from app.api.showcase.models import ShowcaseItem
from attr.setters import pipe
from fedot.core.pipelines.pipeline import Pipeline
from numpy.typing import NDArray


def _assert_arrays(given: List[Any], etalon: List[Any], label: str = 'x'):
    assert len(given) == len(etalon), (
        f'{label} length error: len({label})={len(given)} != len(etalon_{label})={len(etalon)}'
    )
    if given and type(given[0]) is list:
        for idx, (giveni, etaloni) in enumerate(zip(given, etalon)):
            assert giveni == pytest.approx(etaloni), (
                f'Incorrect {label}{idx} mapping: {label}{idx}={giveni} != etalon_{label}{idx}={etaloni}'
            )
    else:
        assert given == pytest.approx(etalon), f'Incorrect {label} mapping: {label}={given} != etalon_{label}={etalon}'


@dataclass
class ChartInputCase:
    x: List[int]
    ys: List[List[Union[int, float]]]
    x_title: str
    y_title: str
    names: List[str]
    y_bnd: Optional[Tuple[int, int]]


MAKE_CHART_DICTS_INPUTS = [
    ChartInputCase(
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
    ChartInputCase(
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
    ChartInputCase(
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


@pytest.mark.parametrize('case', MAKE_CHART_DICTS_INPUTS)
def test_make_chart_dicts_for_boxplot(case: ChartInputCase):
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
    assert result == etalon, f'Chart dicts aren\'t equal: {result} != {etalon}'


@pytest.mark.parametrize('case', MAKE_CHART_DICTS_INPUTS)
def test_make_chart_dicts(case: ChartInputCase):
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
    assert result_series == etalon_series, f'Series aren\'t equal: {result_series} != {etalon_series}'
    assert result_options == etalon_options, f'Options aren\'t equal: {result_options} != {etalon_options}'


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
    etalon_x = [0, 1, 2]
    _assert_arrays(x, etalon_x, 'x')
    etalon_y = [1.111, 3.337, 6.123]
    _assert_arrays(y, etalon_y, 'y')


@pytest.fixture
def box_plot_data_fixture(monkeypatch):
    @dataclass
    class MockBoxPlotData:
        series: Tuple[list, list]

        def __iter__(self):
            return iter(self.series)

    monkeypatch.setattr('app.api.analytics.service.BoxPlotData', MockBoxPlotData)


@dataclass
class PopulationAnalytics:
    analytic_type: str
    etalon_x: List[int]
    etalon_y: List[Any]


GET_POPULATION_ANALYTICS_INPUTS = [
    PopulationAnalytics(
        analytic_type='pheno',
        etalon_x=[0, 1, 2],
        etalon_y=[[1.1111, 2.2229999], [3.3366], [5.54321, 6.123456, 10.54346]]
    ),
    PopulationAnalytics(
        analytic_type='geno',
        etalon_x=[0, 1, 2],
        etalon_y=[[1, 4], [9], [25, 36, 100]]
    ),
    PopulationAnalytics(
        analytic_type='exception',
        etalon_x=[],
        etalon_y=[]
    )
]


@pytest.mark.parametrize('case', GET_POPULATION_ANALYTICS_INPUTS)
def test_get_population_analytics(
    case: PopulationAnalytics,
    monkeypatch,
    composer_history_for_case_fixture,
    chart_dicts_fixture,
    box_plot_data_fixture
):
    if case.analytic_type == 'exception' and not case.etalon_x and not case.etalon_y:
        with pytest.raises(ValueError):
            get_population_analytics('', case.analytic_type)
    else:
        x, y = get_population_analytics('', case.analytic_type)
        assert x and y, 'MockBoxPlotData is empty'
        assert len(x) == len(y), 'x and y should have the same shape'
        _assert_arrays(x, case.etalon_x, 'x')
        _assert_arrays(y, case.etalon_y, 'y')


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
    predict: NDArray = None


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
    is_base: bool = False  # non-existing field

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
class PredictionForPipeline:
    showcase: MockShowcaseItem
    pipeline: Optional[MockPipeline]
    target: Union[str, Callable[[Any], Any]]


TEST_PREDICTION_FOR_PIPELINE_INPUTS = [
    PredictionForPipeline(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=None,
        target=lambda test_data, prediction: test_data is None and prediction is None
    ),
    PredictionForPipeline(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=MockPipeline(),
        target='exception'
    ),
    PredictionForPipeline(
        showcase=MockShowcaseItem(MockMetaData(dataset_name=None)),
        pipeline=MockPipeline(is_fitted=True),
        target=lambda test_data, prediction: test_data is None and prediction is None
    ),
    PredictionForPipeline(
        showcase=MockShowcaseItem(),
        pipeline=None,
        target=lambda test_data, prediction: type(test_data) is MockInputData and prediction is None
    ),
    PredictionForPipeline(
        showcase=MockShowcaseItem(),
        pipeline=MockPipeline(),
        target=lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
    ),
    PredictionForPipeline(
        showcase=MockShowcaseItem(),
        pipeline=MockPipeline(is_fitted=True),
        target=lambda test_data, prediction: type(test_data) is MockInputData and type(prediction) is MockOutputData
    ),
]


@pytest.mark.parametrize('case', TEST_PREDICTION_FOR_PIPELINE_INPUTS)
def test_test_prediction_for_pipeline(
    case: PredictionForPipeline,
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
            _test_prediction_for_pipeline(case.showcase, case.pipeline)
    else:
        case.target(*_test_prediction_for_pipeline(case.showcase, case.pipeline))


@dataclass
class ModellingResults:
    task_name: str
    pipeline: Optional[MockPipeline]
    baseline_pipeline: Optional[MockPipeline]
    etalon_x: List[int]
    etalon_y: List[Any]


GET_MODELLING_RESULTS_INPUTS = [
    ModellingResults(
        task_name='classification',
        pipeline=None,
        baseline_pipeline=None,
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='classification',
        pipeline=None,
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='classification',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        etalon_x=[0, 1, 2, 3, 4, 5],
        etalon_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModellingResults(
        task_name='classification',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[0, 1, 2, 3, 4, 5],
        etalon_y=[[1, 2, 3, 4, 5, 6], [22, 33, 44, 55, 66, 77]]
    ),

    ModellingResults(
        task_name='regression',
        pipeline=None,
        baseline_pipeline=None,
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='regression',
        pipeline=None,
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='regression',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        etalon_x=[0, 1, 2, 3, 4, 5],
        etalon_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModellingResults(
        task_name='regression',
        pipeline=MockPipeline,
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[0, 1, 2, 3, 4, 5],
        etalon_y=[[1, 2, 3, 4, 5, 6], [22, 33, 44, 55, 66, 77]]
    ),

    ModellingResults(
        task_name='ts_forecasting',
        pipeline=None,
        baseline_pipeline=None,
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='ts_forecasting',
        pipeline=None,
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='ts_forecasting',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        etalon_x=[0],
        etalon_y=[[1, 2, 3, 4, 5, 6]]
    ),
    ModellingResults(
        task_name='ts_forecasting',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[0],
        etalon_y=[[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]
    ),

    ModellingResults(
        task_name='exception',
        pipeline=None,
        baseline_pipeline=None,
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='exception',
        pipeline=None,
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='exception',
        pipeline=MockPipeline(),
        baseline_pipeline=None,
        etalon_x=[],
        etalon_y=[]
    ),
    ModellingResults(
        task_name='exception',
        pipeline=MockPipeline(),
        baseline_pipeline=MockPipeline(is_base=True),
        etalon_x=[],
        etalon_y=[]
    ),
]


@pytest.mark.parametrize('case', GET_MODELLING_RESULTS_INPUTS)
def test_get_modelling_results(
    case: ModellingResults,
    monkeypatch,
    plot_data_fixture,
    chart_dicts_fixture,
    showcase_item_fixture
):
    @dataclass
    class MockPipeline:  # using non-existing fields
        is_base: bool = False
        pass
    monkeypatch.setattr(
        'app.api.analytics.service.Pipeline', MockPipeline
    )

    def mock_test_prediction_for_pipeline(showcase: MockShowcaseItem, pipeline: MockPipeline, *args, **kwargs):
        if pipeline:
            if pipeline.is_base:
                if showcase.metadata.task_name == 'ts_forecasting':
                    return None, MockOutputData(np.array([[10, 11, 12, 13, 14, 15]]))
                return None, MockOutputData(np.array([[22], [33], [44], [55], [66], [77]]))
            else:
                if showcase.metadata.task_name == 'ts_forecasting':
                    return None, MockOutputData(np.array([[1, 2, 3, 4, 5, 6]]))
                return None, MockOutputData(np.array([[1], [2], [3], [4], [5], [6]]))
        return None, None
    monkeypatch.setattr(
        'app.api.analytics.service._test_prediction_for_pipeline', mock_test_prediction_for_pipeline
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
        _assert_arrays(x, case.etalon_x, 'x')
        _assert_arrays(y, case.etalon_y, 'y')
