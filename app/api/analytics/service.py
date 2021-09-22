from typing import Any, Dict, List, Optional, Tuple, Union

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


def _process_y_value(y):
    y_new = y
    if isinstance(y, list) or isinstance(y, np.ndarray):
        y_new = y[0]
    return round(y_new, 3)


def _make_chart_dicts_for_boxplot(
    x: List[int],
    ys: List[List[Integral]],
    x_title: str, y_title: str  # TODO: are these needed?
) -> List[Dict]:
    return [
        {
            'x': [round(_, 3) for _ in y],
            'y': f'Gen {x[idx]}',
            'type': 'box',
            'name': idx
        }
        for idx, y in enumerate(ys)
    ]


def _make_chart_dicts(
    x: List[int],
    ys: List[List[Integral]],
    names: List[str],
    x_title: str, y_title: str, plot_type: str,
    y_bnd: Optional[Tuple[int, int]] = None
) -> Tuple[List[Dict], Dict]:
    series: List[Dict] = [
        {
            'name': names[idx1],
            'data':
                [round(_, 3) for _ in y]
                if plot_type == 'line'
                else [[x[idx2], round(_, 3)] for idx2, _ in enumerate(y)]
        }
        for idx1, y in enumerate(ys)
    ]

    if not y_bnd:
        min_y = min(_process_y_value(min(y)) for y in ys) * 0.95
        max_y = max(_process_y_value(max(y)) for y in ys) * 1.05
    else:
        min_y, max_y = y_bnd

    options: Dict = {
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
    history = composer_history_for_case(case_id)

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
    history = composer_history_for_case(case_id)

    y_gen: List[List[Integral]]
    if analytic_type == 'pheno':
        y_gen = [[abs(i.fitness) for i in gen] for gen in history.individuals]
    elif analytic_type == 'geno':
        y_gen = [[abs(i.graph.depth) for i in gen] for gen in history.individuals]
    else:
        raise ValueError(f'Analytic type {analytic_type} not recognized')

    x: List[int] = [idx for idx, _ in enumerate(history.individuals)]

    series: List[Dict] = _make_chart_dicts_for_boxplot(
        x=x, ys=y_gen,
        x_title='Epochs', y_title='Fitness'
    )

    output = BoxPlotData(series=series)
    return output


def get_prediction_for_pipeline(
    case: ShowcaseItem, pipeline: Optional[Pipeline]
) -> Tuple[Optional[InputData], Optional[OutputData]]:
    test_data = get_input_data(dataset_name=case.metadata.dataset_name, sample_type='test')
    prediction: Optional[OutputData] = None
    if pipeline:
        train_data: Optional[InputData] = get_input_data(
            dataset_name=case.metadata.dataset_name, sample_type='train'
        )
        if not pipeline.is_fitted and train_data:
            pipeline.fit(train_data)
        prediction = pipeline.predict(test_data)
    return test_data, prediction


def get_modelling_results(
    case: ShowcaseItem,
    pipeline: Optional[Pipeline],
    baseline_pipeline: Optional[Pipeline] = None
) -> PlotData:
    test_data, prediction = get_prediction_for_pipeline(case, pipeline)

    y_bnd: Optional[Tuple[int, int]] = None
    x_title: str
    y_title: str
    case_task_name: str = case.metadata.task_name
    if case_task_name == 'classification':
        x_title, y_title = 'Item', 'Probability'
        y_bnd = (0, 1)
    elif case_task_name == 'regression':
        x_title, y_title = 'Item', 'Value'
    elif case_task_name == 'ts_forecasting':
        x_title, y_title = 'Time step', 'Value'
    else:
        raise NotImplementedError(f'Task {case_task_name} not supported')

    plot_type: str
    if case_task_name == 'ts_forecasting':
        plot_type = 'line'
    else:
        plot_type = 'scatter'
    y: List[Integral] = prediction.predict.ravel().astype(float).tolist()
    baseline_prediction: Optional[OutputData] = None
    if baseline_pipeline:
        _, baseline_prediction = get_prediction_for_pipeline(case, baseline_pipeline)
    y_baseline: Optional[List[Integral]] = (
        baseline_prediction.predict.ravel().astype(float).tolist() if baseline_prediction else None
    )
    y_obs: Optional[List[Integral]] = (
        test_data.target.ravel().astype(float).tolist() if getattr(test_data, 'target', None) is not None else None
    )

    x: List[int] = [idx for idx, _ in enumerate(prediction.predict[:max_items_in_plot])]
    y = y[:max_items_in_plot]
    ys: List[List[Integral]] = [y]
    names: List[str] = ['Candidate']
    if baseline_prediction:
        y_baseline = y_baseline[:max_items_in_plot]
        ys.append(y_baseline)
        names.append('Baseline')
    if y_obs is not None:
        y_obs = y_obs[:max_items_in_plot]
        ys.append(y_obs)
        names.append('Observations')

    series, options = _make_chart_dicts(x=x, ys=ys, names=names,
                                        x_title=x_title, y_title=y_title,
                                        plot_type=plot_type, y_bnd=y_bnd)

    return PlotData(series, options)
