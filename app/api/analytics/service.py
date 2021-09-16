import typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import numpy as np
from app.api.composer.service import composer_history_for_case
from app.api.data.service import get_input_data
from app.api.showcase.models import ShowcaseItem
from fedot.core.data.data import InputData, OutputData
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline

from .models import BoxPlotData, PlotData

max_items_in_plot: int = 50

# TODO: make typings file
Integral = Union[int, float]


def _make_chart_dicts_for_boxplot(
    x: List[int],
    ys: List[List[Integral]],
    x_title: str, y_title: str  # TODO: are these needed?
) -> List[Dict[str, Union[str, int, List[Integral]]]]:
    return [
        {
            'x': [round(_, 3) for _ in y],
            'y': f'Gen {x[idx]}',
            'type': 'box',
            'name': idx
        }
        for idx, y in enumerate(ys)
    ]


def _process_y_value(y):
    y_new = y
    if isinstance(y, list) or isinstance(y, np.ndarray):
        y_new = y[0]
    return round(y_new, 3)


def _make_chart_dicts(
    x: List[int],
    ys: List[List[Integral]],
    names: List[str],
    x_title: str, y_title: str, plot_type: str, y_bnd=None
) -> Tuple[
    List[Dict[str, Union[str, List[Integral], List[List[Union[int, Integral]]]]]],
    Dict[str, Dict[str, Any]]
]:
    series: List[Dict[
        str,
        Union[str, List[Integral], List[List[Union[int, Integral]]]]
    ]] = [
        {
            'name': names[idx1],
            'data':
                [round(_, 3) for _ in y]
                if plot_type == "line"
                else [[x[idx2], round(_, 3)] for idx2, _ in enumerate(y)]
        }
        for idx1, y in enumerate(ys)
    ]

    if not y_bnd:
        min_y: float = min(min(y) for y in ys) * 0.95
        max_y: float = max(max(y) for y in ys) * 1.05
    else:
        min_y, max_y = y_bnd

    options: Dict[str, Dict[str, Any]] = {
        'chart': {
            'type': plot_type
        },
        'xaxis': {
            'categories': x,
            'title': {
                'text': x_title
            }
        },
        'yaxis': {
            'title': {
                'text': y_title
            },
            'min': min_y,
            'max': max_y
        }
    }
    return series, options


def get_quality_analytics(case_id: str) -> PlotData:
    history: OptHistory = composer_history_for_case(case_id)

    y: List[Integral] = [
        round(abs(min(i.fitness for i in gen)), 3)
        for gen in history.individuals
    ]
    x: List[int] = [idx for idx, _ in enumerate(history.individuals)]

    series, options = _make_chart_dicts(
        x=x, ys=[y],
        names=['Test sample'],
        x_title='Epochs', y_title='Fitness', plot_type='line'
    )

    output = PlotData(series=series, options=options)
    return output


def get_population_analytics(case_id: str, analytic_type: str) -> BoxPlotData:
    history: OptHistory = composer_history_for_case(case_id)

    y_gen: List[List[Integral]]
    if analytic_type == 'pheno':
        y_gen = [[abs(i.fitness) for i in gen] for gen in history.individuals]
    elif analytic_type == 'geno':
        y_gen = [[abs(i.graph.depth) for i in gen] for gen in history.individuals]
    else:
        raise ValueError(f'Analytic type {analytic_type} not recognized')

    x: List[int] = [idx for idx, _ in enumerate(history.individuals)]

    series: List[Dict[str, Union[str, int, List[Integral]]]] = _make_chart_dicts_for_boxplot(
        x=x, ys=y_gen,
        x_title='Epochs', y_title='Fitness'
    )

    output = BoxPlotData(series=series)
    return output


def _test_prediction_for_pipeline(
    case: ShowcaseItem, pipeline: Optional[Pipeline]
) -> Tuple[Optional[InputData], Optional[OutputData]]:
    test_data: Optional[InputData] = get_input_data(dataset_name=case.metadata.dataset_name, sample_type='test')
    prediction: Optional[OutputData] = None
    if pipeline:
        train_data: Optional[InputData]
        if not pipeline.is_fitted and (train_data := get_input_data(
            dataset_name=case.metadata.dataset_name, sample_type='train'
        )):
            pipeline.fit(train_data)
        prediction = pipeline.predict(test_data)
    return test_data, prediction


def get_modelling_results(
    case: ShowcaseItem,
    pipeline: Optional[Pipeline],
    baseline_pipeline: Optional[Pipeline] = None
) -> PlotData:
    _, prediction = _test_prediction_for_pipeline(case, pipeline)
    baseline_prediction: Optional[OutputData] = None
    if baseline_pipeline:
        _, baseline_prediction = _test_prediction_for_pipeline(case, baseline_pipeline)
    y_bnd: Optional[Tuple[int, ...]] = None
    x_title: str
    y_title: str
    if (case_task_name := case.metadata.task_name) == 'classification':
        x_title, y_title = 'Item', 'Probability'
        y_bnd = (0, 1)
    elif case_task_name == 'regression':
        x_title, y_title = 'Item', 'Value'
    elif case_task_name == 'ts_forecasting':
        x_title, y_title = 'Time step', 'Value'
    else:
        raise NotImplementedError(f'Task {case.metadata.task_name} not supported')

    if case.metadata.task_name == 'ts_forecasting':
        plot_type = 'line'
        y = list(prediction.predict[0, :])
        y_baseline = list(baseline_prediction.predict[0, :]) if baseline_prediction else None
        y_obs = list([float(o) for o in obs]) if obs is not None else None

    else:
        plot_type = 'scatter'
        y = prediction.predict.tolist()
        if baseline_prediction:
            y_baseline = list(baseline_prediction.predict) if baseline_prediction else None
        y_obs = obs

    x: List[int] = [idx for idx, _ in enumerate(prediction.predict[:max_items_in_plot])]
    y = y[:max_items_in_plot]
    ys: List[List[Integral]] = [y]
    names: List[str] = ['Candidate']
    if baseline_prediction:
        y_baseline = y_baseline[:max_items_in_plot]
        ys.append(y_baseline)
        names.append('Baseline')
    if obs is not None:
        y_obs = y_obs[:max_items_in_plot]
        ys.append(y_obs)
        names.append('Observations')

    series, options = _make_chart_dicts(x=x, ys=ys, names=names,
                                        x_title=x_title, y_title=y_title,
                                        plot_type=plot_type, y_bnd=y_bnd)

    return PlotData(series, options)
