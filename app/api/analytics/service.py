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
    x_title: str, y_title: str
) -> List[Dict[str, Union[str, int, list]]]:
    series: List[Dict[str, Union[str, int, list]]] = []

    for i in range(len(ys)):
        y: List[int] = [round(_, 3) for _ in ys[i]]
        series.append({
            'y': f'Gen {x[i]}',
            'x': y,
            'type': 'box',
            'name': i
        })

    return series


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
    List[Dict[str, Union[str, List[int], List[List[int]]]]],
    Dict[str, Dict[str, Any]]
]:
    series: List[Dict[str, Union[str, List[int], List[List[int]]]]] = []

    for i in range(len(ys)):
        if plot_type == 'line':
            data = [round(_, 3) for _ in ys[i]]
        else:
            data = [[x[j], round(_process_y_value(ys[i][j]), 3)] for j in range(len(ys[i]))]

        series.append({
            'name': names[i],
            'data': data
        })

    if not y_bnd:
        min_y = min([_process_y_value(min(y)) for y in ys]) * 0.95
        max_y = max([_process_y_value(max(y)) for y in ys]) * 1.05
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

    y: List[int] = [round(abs(min([i.fitness for i in gen])), 3) for gen in history.individuals]
    x: List[int] = list(range(len(history.individuals)))

    series, options = _make_chart_dicts(x=x, ys=[y], names=['Test sample'],
                                        x_title='Epochs', y_title='Fitness',
                                        plot_type='line')

    output = PlotData(series=series, options=options)
    return output


def get_population_analytics(case_id: str, analytic_type: str) -> BoxPlotData:
    history: OptHistory = composer_history_for_case(case_id)

    if analytic_type == 'pheno':
        y_gen: List[List[Integral]] = [[abs(i.fitness) for i in gen] for gen in history.individuals]
    elif analytic_type == 'geno':
        y_gen: List[List[Integral]] = [[abs(i.graph.depth) for i in gen] for gen in history.individuals]
    else:
        raise ValueError(f'Analytic type {analytic_type} not recognized')

    x: List[int] = list(range(len(history.individuals)))

    series: List[Dict[str, Union[str, int, list]]] = _make_chart_dicts_for_boxplot(x=x, ys=y_gen,
                                           x_title='Epochs', y_title='Fitness')

    output = BoxPlotData(series=series)
    return output


def _test_prediction_for_pipeline(
    case: ShowcaseItem, pipeline: Optional[Pipeline]
) -> Tuple[Optional[InputData], OutputData]:
    train_data: Optional[InputData] = get_input_data(dataset_name=case.metadata.dataset_name, sample_type='train')

    if not pipeline.is_fitted:
        pipeline.fit(train_data)

    test_data: Optional[InputData] = get_input_data(dataset_name=case.metadata.dataset_name, sample_type='test')
    prediction: OutputData = pipeline.predict(test_data)
    return test_data, prediction


def get_modelling_results(
    case: ShowcaseItem,
    pipeline: Optional[Pipeline],
    baseline_pipeline: Optional[Pipeline]=None
) -> PlotData:
    _, prediction = _test_prediction_for_pipeline(case, pipeline)
    baseline_prediction: Optional[OutputData] = None
    if baseline_pipeline:
        _, baseline_prediction = _test_prediction_for_pipeline(case, baseline_pipeline)
    y_bnd: Optional[Tuple[int, ...]] = None
    if case.metadata.task_name == 'classification':
        y_title: str = 'Probability'
        x_title: str = 'Item'
        y_bnd = (0, 1)
    elif case.metadata.task_name == 'regression':
        y_title: str = 'Value'
        x_title: str = 'Item'
    elif case.metadata.task_name == 'ts_forecasting':
        y_title: str = 'Value'
        x_title: str = 'Time step'
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

    x: List[int] = list(range(len(prediction.predict)))
    x = x[:max_items_in_plot]
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
