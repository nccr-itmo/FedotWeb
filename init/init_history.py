import json
import os
from pathlib import Path
from typing import Optional, Union

from app.api.composer.service import run_composer
from app.api.data.service import get_input_data
from app.api.pipelines.service import create_pipeline, is_pipeline_exists
from app.singletons.db_service import DBServiceSingleton
from bson import json_util
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.serializers import json_helpers
from flask import current_app
from utils import project_root

from init.init_pipelines import _extract_pipeline_with_fitted_operations


def create_default_history(opt_times=None):
    if opt_times is None:
        opt_times = [2, 1, 1]

    cases = [
        {
            'history_id': 'scoring', 'dataset_name': 'scoring',
            'metric': 'roc_auc', 'task': 'classification',
            'time': opt_times[0]
        },
        {
            'history_id': 'metocean', 'dataset_name': 'metocean',
            'metric': 'rmse', 'task': 'ts_forecasting',
            'time': opt_times[1]
        },
        {
            'history_id': 'oil', 'dataset_name': 'oil',
            'metric': 'rmse', 'task': 'regression',
            'time': opt_times[2]
        }
    ]

    mock_list = []
    for case in cases:
        candidate_path = Path(f'{project_root()}/data/{case["history_id"]}/{case["history_id"]}_{case["task"]}.json')
        if candidate_path.exists():
            case['external_history'] = candidate_path
        mock_list.append(_init_composer_history_for_case(**case))

    if not DBServiceSingleton().exists():
        mockup_history(mock_list)


def mockup_history(mock_list):
    if len(mock_list) > 0:
        history = [i['history'] for i in mock_list]
        with open(os.path.join(project_root(), 'test/fixtures/history.json'), 'w') as f:
            f.write(json_util.dumps(history))
            print('history are mocked')

        pipelines = [j for i in mock_list for j in i['pipelines_dict']]

        if len(pipelines) > 0:
            with open(os.path.join(project_root(), 'test/fixtures/pipelines.json'), 'r+') as f:
                data = json_util.loads(f.read())
                data.extend(pipelines)
                f.seek(0)
                f.write(json_util.dumps(data))
                print('history pipelines are mocked')

        dicts_fitted_operations = [j for i in mock_list for j in i['dicts_fitted_operations']]
        if len(dicts_fitted_operations) > 0:
            with open(os.path.join(project_root(), 'test/fixtures/dict_fitted_operations.json'), 'r+') as f:
                data = json_util.loads(f.read())
                data.extend(dicts_fitted_operations)
                f.seek(0)
                f.write(json_util.dumps(data))
                print('history dict_fitted_operations are mocked')


def _save_history_to_path(history: OptHistory, path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir()
    path.write_text(json.dumps(history, default=json_helpers.encoder))


def _init_composer_history_for_case(history_id, task, metric, dataset_name, time,
                                    external_history: Optional[Union[dict, os.PathLike]] = None):
    mock_dct = {}

    db_service = DBServiceSingleton()
    history_path = None

    if external_history is None:
        # run composer in real-time
        history = run_composer(task, metric, dataset_name, time)
        history_obj = json.dumps(history, default=json_helpers.encoder)
    elif isinstance(external_history, dict):
        # init from dict
        history_obj = external_history
        history = json.loads(json.dumps(external_history, default=json_helpers.encoder),
                             object_hook=json_helpers.decoder)
    else:
        # load from path
        history_path = Path(external_history)
        history = run_composer(task, metric, dataset_name, time, history_path)
        history_obj = json.dumps(history, default=json_helpers.encoder)

    if history_path is None:
        history_path = Path(f'{project_root()}/data/{history_id}/{history_id}_{task}.json')

    _save_history_to_path(history, history_path)

    if db_service.exists():
        if current_app and current_app.config['CONFIG_NAME'] == 'test':
            db_service.try_reinsert_one('history', {'history_id': history_id}, history_obj)
        else:
            db_service.try_reinsert_file({'filename': history_id, 'type': 'history'}, history_obj)
    else:
        history_obj = {
            'history_id': history_id,
            'history_json': history_obj
        }

    mock_dct['history'] = history_obj
    mock_dct['pipelines_dict'] = []
    mock_dct['dicts_fitted_operations'] = []

    data = get_input_data(dataset_name=dataset_name, sample_type='train')

    best_fitness = None

    for i, pipeline_template in enumerate(history.historical_pipelines):
        pipeline_uid = pipeline_template.unique_pipeline_id

        fitness = history.all_historical_fitness[i]
        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness

            case = db_service.try_find_one('cases', {'case_id': history_id})
            if case is not None:
                case['pipeline_id'] = pipeline_uid
                db_service.try_reinsert_one('cases', {'case_id': history_id}, case)

        is_existing_pipeline = is_pipeline_exists(pipeline_uid)
        if not is_existing_pipeline:
            print(i)
            pipeline = Pipeline()
            pipeline_template.convert_to_pipeline(pipeline)
            pipeline.fit(data)
            if db_service.exists():
                create_pipeline(pipeline_uid, pipeline)
            else:
                pipeline_dict, dict_fitted_operations = _extract_pipeline_with_fitted_operations(pipeline, pipeline_uid)
                mock_dct['pipelines_dict'].append(pipeline_dict)
                mock_dct['dicts_fitted_operations'].append(dict_fitted_operations)

    return mock_dct
