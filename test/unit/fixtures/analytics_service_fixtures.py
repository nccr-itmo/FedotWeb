from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from ..mocks.analytics_service_mocks import (MockIndividual,
                                             MockIndividualGraph,
                                             MockInputData, MockOptHistory,
                                             MockOutputData, MockPipeline,
                                             MockShowcaseItem)


@pytest.fixture
def plot_data_fixture(monkeypatch):
    @dataclass
    class MockPlotData:
        series: list
        options: list

        def __iter__(self):
            return iter((self.series, self.options))

    monkeypatch.setattr('app.api.analytics.service.PlotData', MockPlotData)


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

    def mock_composer_history_for_case(case_id: str, *args, **kwargs):
        return MockOptHistory(individuals)
    monkeypatch.setattr(
        'app.api.analytics.service.composer_history_for_case', mock_composer_history_for_case
    )


@pytest.fixture
def make_chart_dicts_fixture(monkeypatch):
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


@pytest.fixture
def box_plot_data_fixture(monkeypatch):
    @dataclass
    class MockBoxPlotData:
        series: Tuple[list, list]

        def __iter__(self):
            return iter(self.series)

    monkeypatch.setattr('app.api.analytics.service.BoxPlotData', MockBoxPlotData)


@pytest.fixture
def showcase_item_fixture(monkeypatch):
    monkeypatch.setattr(
        'app.api.analytics.service.ShowcaseItem', MockShowcaseItem
    )


@pytest.fixture
def input_output_data_fixture(monkeypatch):
    def mock_get_input_data(*args, **kwargs):
        if kwargs['dataset_name'] is None:
            return None
        return MockInputData()
    monkeypatch.setattr(
        'app.api.analytics.service.get_input_data', mock_get_input_data
    )
    monkeypatch.setattr(
        'app.api.analytics.service.InputData', MockInputData
    )
    monkeypatch.setattr(
        'app.api.analytics.service.OutputData', MockOutputData
    )


@pytest.fixture
def pipeline_fixture(monkeypatch):
    monkeypatch.setattr(
        'app.api.analytics.service.Pipeline', MockPipeline
    )


@pytest.fixture
def pipeline_prediction_fixture(monkeypatch):
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
