import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import gridfs
import pymongo
from bson import json_util
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from flask import current_app, has_app_context, url_for
from pymongo.errors import DuplicateKeyError

from app import storage
from utils import project_root


def is_pipeline_exists(db, uid: str) -> Optional[Pipeline]:
    pipeline_dict = db.pipelines.find_one({'uid': str(uid)})
    return pipeline_dict is not None


def pipeline_by_uid(uid: str) -> Optional[Pipeline]:
    pipeline = Pipeline()
    pipeline_dict = storage.db.pipelines.find_one({'uid': str(uid)})
    if pipeline_dict is None:
        return None

    dict_fitted_operations = None
    if current_app.config['CONFIG_NAME'] == 'test':
        dict_fitted_operations = storage.db.dict_fitted_operations.find_one({'uid': str(uid)})
    else:
        try:
            fs = gridfs.GridFS(storage.db)
            file = fs.find_one({'filename': str(uid), 'type': 'dict_fitted_operations'}).read()
            dict_fitted_operations = json_util.loads(file)
        except AttributeError as ex:
            print(f'dict_fitted_operations not found for {uid}')

    if dict_fitted_operations:
        for key in dict_fitted_operations:
            if key.find('operation') != -1:
                bytes_container = BytesIO()
                bytes_container.write(dict_fitted_operations[key])
                dict_fitted_operations[key] = bytes_container
    pipeline.load(pipeline_dict, dict_fitted_operations)
    return pipeline


def validate_pipeline(pipeline: Pipeline) -> Tuple[bool, str]:
    try:
        validate(pipeline)
        return True, 'Correct pipeline'
    except ValueError as ex:
        return False, str(ex)


def create_pipeline(db, uid: str, pipeline: Pipeline, overwrite=False):
    is_new = True
    existing_uid = db.pipelines.find_one({'uid': str(uid)})
    if existing_uid and not overwrite:
        is_new = False

    dumped_json, dict_fitted_operations = pipeline.save()
    if dict_fitted_operations:
        for key in dict_fitted_operations:
            if key.find('operation') != -1:
                dict_fitted_operations[key].seek(0)
                saved_operation = dict_fitted_operations[key].read()
                dict_fitted_operations[key] = saved_operation
        dict_fitted_operations['uid'] = str(uid)

    if is_new:
        dict_pipeline = json.loads(dumped_json)
        _add_pipeline_to_db(db, uid, dict_pipeline, dict_fitted_operations, overwrite=overwrite)
    else:
        warnings.warn('Cannot create new pipeline')

    return uid, is_new


def _add_pipeline_to_db(db, uid, dict_pipeline, dict_fitted_operations, init_db=False, overwrite=False):
    dict_pipeline['uid'] = str(uid)
    if init_db:
        db.pipelines.remove({'uid': uid})
    else:
        is_exists = db.pipelines.find_one({'uid': uid}) is not None
        if is_exists and overwrite:
            db.pipelines.remove({'uid': uid})
            is_exists = False
        if not is_exists:
            try:
                db.pipelines.insert_one(dict_pipeline)
            except DuplicateKeyError as ex:
                print(f'Suddenly, pipeline {str(uid)} already exists: {ex}')

    if dict_fitted_operations is not None:
        try:
            if has_app_context() and current_app.config['CONFIG_NAME'] == 'test':
                dict_fitted_operations = storage.db.dict_fitted_operations.find_one({'uid': str(uid)})
            else:
                fs = gridfs.GridFS(db)
                file = fs.find_one({'filename': uid, 'type': 'dict_fitted_operations'})
                if file:
                    fs.delete(file._id)
                fs.put(json_util.dumps(dict_fitted_operations), filename=uid, type='dict_fitted_operations',
                       encoding='utf-8')

            if init_db:
                return [dict_pipeline, dict_fitted_operations]

        except DuplicateKeyError:
            print(f'Fitted operations dict {str(uid)} already exists')
        except pymongo.errors.DocumentTooLarge as ex:
            print(f'Dict {str(uid)} too large: {ex}')


def get_image_url(filename, pipeline):
    image_path = f'{project_root()}/frontend/build/static/generated_images/{filename}'
    image = Path(image_path)
    if not image.exists():
        dir_path = Path(f'{project_root()}/frontend/build/static/generated_images/')
        if not dir_path.exists():
            dir_path.mkdir()
        pipeline.show(image_path)
    return url_for('static', filename=f'generated_images/{filename}')


def get_pipeline_metadata(pipeline_id) -> Tuple[int, int]:
    pipeline = pipeline_by_uid(pipeline_id)
    if not pipeline:
        return -1, -1
    return pipeline.length, pipeline.depth


def _replace_symbols_in_dct_keys(dct, old, new_symb):
    for key in list(dct.keys()):
        new_key = key.replace(old, new_symb)
        dct[new_key] = dct.pop(key)
    return dct
