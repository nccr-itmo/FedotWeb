from dataclasses import dataclass
from typing import Tuple

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
