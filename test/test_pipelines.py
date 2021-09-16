import json
import os
from pathlib import Path

from app.api.pipelines.pipeline_convert_utils import pipeline_to_graph
from app.api.pipelines.service import get_image_url
from init.init_pipelines import pipeline_mock
from utils import project_root


def test_get_pipeline_endpoint(client):
    uid = 'best_scoring_pipeline'
    pipeline_json = client.get(f'api/pipelines/{uid}').json
    assert pipeline_json['uid'] == uid
    assert len(pipeline_json['nodes']) > 0
    assert len(pipeline_json['edges']) > 0

    knn_node = [node for node in pipeline_json['nodes'] if node['model_name'] == 'knn'][0]
    assert knn_node['type'] == 'model'

    pca_node = [node for node in pipeline_json['nodes'] if node['model_name'] == 'pca'][0]
    assert pca_node['type'] == 'data_operation'


def test_validate_pipeline_endpoint(client):
    pipeline = pipeline_mock()
    graph = pipeline_to_graph(pipeline)

    validation_results = client.post('api/pipelines/validate', json=graph).json
    assert validation_results['is_valid'] is True


def test_add_pipeline_endpoint(client):
    with open(os.path.join(project_root(), 'test', 'data', 'graph_example.json')) as f:
        graph = json.load(f)

    basic_uid = 'new_pipeline'
    graph = {
        'nodes': graph['nodes'],
        'edges': graph['edges']

    }
    response = client.post('api/pipelines/add', json=graph).json
    assert response['is_new'] is True
    assert response['uid'] != basic_uid


def test_pipeline_image_endpoint(client):
    uid = 'best_scoring_pipeline'
    response = client.get(f'api/pipelines/image/{uid}').json
    test_pipeline = pipeline_mock()
    filename = f'{uid}.png'
    image_url = get_image_url(filename, test_pipeline)
    assert Path(f'{project_root()}/frontend/build/static/generated_images/{filename}').exists()
    assert image_url == response['image_url']
