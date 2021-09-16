import pickle

import gridfs
from bson import json_util
from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from flask import current_app

from app import storage
from app.api.data.service import get_input_data
from app.api.pipelines.service import create_pipeline, is_pipeline_exists
from app.api.showcase.models import ShowcaseItem
from app.api.showcase.showcase_utils import prepare_icon_path
from utils import project_root


def showcase_item_from_db(case_id: str) -> ShowcaseItem:
    dumped_item = storage.db.cases.find_one({'case_id': case_id})
    icon_path = prepare_icon_path(dumped_item)
    item = ShowcaseItem(case_id=dumped_item['case_id'],
                        title=dumped_item['title'],
                        icon_path=icon_path,
                        description=dumped_item['description'],
                        pipeline_id=dumped_item['pipeline_id'],
                        metadata=pickle.loads(dumped_item['metadata']))
    return item


def composer_history_for_case(case_id: str, validate_history: bool = False) -> OptHistory:
    case = showcase_item_from_db(case_id)
    task = case.metadata.task_name
    metric = case.metadata.metric_name
    dataset_name = case.metadata.dataset_name

    if current_app.config['CONFIG_NAME'] == 'test':
        saved_history = storage.db.history.find_one({'history_id': case_id})
    else:
        fs = gridfs.GridFS(storage.db)
        file = fs.find_one({'filename': case_id, 'type': 'history'}).read()
        saved_history = json_util.loads(file)

    if not saved_history:
        history = run_composer(task, metric, dataset_name, time=1.0)
        _save_to_db(storage.db, case_id, history)
    else:
        if current_app.config['CONFIG_NAME'] == 'test':
            history = pickle.loads(saved_history['history_pkl'])
        else:
            history = pickle.loads(saved_history)

    data = get_input_data(dataset_name=dataset_name, sample_type='train')

    if validate_history:
        for i, pipeline_template in enumerate(history.historical_pipelines):
            struct_id = pipeline_template.unique_pipeline_id
            existing_pipeline = is_pipeline_exists(storage.db, struct_id)
            if not existing_pipeline:
                print(i)
                pipeline = Pipeline()
                pipeline_template.convert_to_pipeline(pipeline)
                pipeline.fit(data)
                create_pipeline(storage.db, struct_id, pipeline)

    return history


def _save_to_db(db, history_id, history):
    history_obj = {
        'history_id': history_id,
        'history_pkl': pickle.dumps(history)
    }
    _add_to_db(db, 'history_id', history_id, history_obj)


def run_composer(task, metric, dataset_name, time):
    pop_size = 10
    num_of_generations = 5
    learning_time = time

    auto_model = Fedot(problem=task, seed=42, preset='light_steady_state', verbose_level=4,
                       timeout=learning_time,
                       composer_params={'composer_metric': metric,
                                        'pop_size': pop_size,
                                        'num_of_generations': num_of_generations,
                                        'max_arity': 3,
                                        'max_depth': 5})
    auto_model.fit(features=f'{project_root()}/data/{dataset_name}/{dataset_name}_train.csv',
                   target='target')
    history = auto_model.history
    unfit_history(history)
    return history


def unfit_history(history):
    for pop in history.individuals:
        for ind in pop:
            ind.graph.link_to_empty_pipeline.unfit()
    for pop in history.archive_history:
        for ind in pop:
            ind.graph.link_to_empty_pipeline.unfit()

    for ops in history.parent_operators:
        for op in ops:
            for op_part in op:
                for pip in op_part.parent_objects:
                    pip.link_to_empty_pipeline.unfit()


def _add_to_db(db, id_name, id_value, obj_to_add):
    db.history.remove({id_name: id_value})
    db.history.insert_one(obj_to_add)
