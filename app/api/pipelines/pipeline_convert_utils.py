import json

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository

from .models import PipelineGraph


def graph_to_pipeline(graph: dict):
    graph_nodes = graph['nodes']

    pipeline_nodes = []
    for graph_node in graph_nodes:
        pipeline_node = _graph_node_to_pipeline_node(graph_node, graph_nodes, pipeline_nodes)
        pipeline_nodes = _add_to_pipeline_if_necessary(pipeline_node, pipeline_nodes)

    pipeline = Pipeline(pipeline_nodes)

    return pipeline


def pipeline_to_graph(pipeline):
    nodes = []
    edges = []

    local_id = 0
    for pipeline_node in pipeline.nodes:
        node = dict()

        node['id'] = local_id
        pipeline_node.tmp_id = local_id
        node['display_name'] = pipeline_node.operation.operation_type
        node['model_name'] = str(pipeline_node.operation)
        node['params'] = pipeline_node.custom_params
        node = replace_deprecated_values(node)
        node['pipeline_node'] = pipeline_node
        node['type'] = _get_node_type_for_model(pipeline_node.operation.operation_type)

        nodes.append(node)
        local_id += 1

    for node in nodes:
        node['parents'] = []
        node['children'] = []
        if node['pipeline_node'].nodes_from is not None:
            for pipeline_node_parent in node['pipeline_node'].nodes_from:
                # fill parents field
                node['parents'].append(pipeline_node_parent.tmp_id)

                # create edge
                edge = dict()
                edge['source'] = pipeline_node_parent.tmp_id
                edge['target'] = node['id']
                edges.append(edge)

            childs = pipeline.operator.node_children(node['pipeline_node'])
            if childs is not None:
                # fill childs field
                for pipeline_node_child in childs:
                    node['children'].append(pipeline_node_child.tmp_id)
        del node['pipeline_node']

    output_graph = PipelineGraph(uid='', nodes=nodes, edges=edges)

    return output_graph


def replace_deprecated_values(graph_node: dict, depr_values: list = ['Infinity', 'NaN']):
    txt = json.dumps(graph_node)
    for val in depr_values:
        txt = txt.replace(val, 'null')
    graph_node = json.loads(txt)
    return graph_node


def _graph_node_to_pipeline_node(graph_node: dict, existing_graph_nodes: dict, pipeline_nodes):
    is_primary = len(graph_node['parents']) == 0
    if is_primary:
        pipeline_node = PrimaryNode(graph_node['model_name'])
    else:
        parent_pipeline_nodes = []
        for parent_id in graph_node['parents']:
            parent_node = [n for n in existing_graph_nodes if n['id'] == parent_id]

            if len(parent_node) == 0:
                continue
            else:
                parent_node = parent_node[0]

            parent_pipeline_node = _graph_node_to_pipeline_node(parent_node, existing_graph_nodes, pipeline_nodes)

            # check is parent node already created
            existing_pipeline_node = _get_identical_pipeline_node(parent_pipeline_node, pipeline_nodes)
            if existing_pipeline_node is not None:
                parent_pipeline_node = existing_pipeline_node

            parent_pipeline_nodes.append(parent_pipeline_node)
            pipeline_nodes = _add_to_pipeline_if_necessary(parent_pipeline_node, pipeline_nodes)

        pipeline_node = SecondaryNode(graph_node['model_name'], nodes_from=parent_pipeline_nodes)

    if graph_node['params'] != 'default_params':
        pipeline_node.custom_params = graph_node['params']

    return pipeline_node


def _add_to_pipeline_if_necessary(new_node, pipeline_nodes):
    pipeline_nodes_ids = [_.descriptive_id for _ in pipeline_nodes]
    if new_node.descriptive_id not in pipeline_nodes_ids:
        pipeline_nodes.append(new_node)
    return pipeline_nodes


def _get_identical_pipeline_node(node, pipeline_nodes):
    try:
        return [_ for _ in pipeline_nodes if _.descriptive_id == node.descriptive_id][0]
    except IndexError:
        return None


def _get_node_type_for_model(model_type: str):
    data_operations = OperationTypesRepository(operation_type='data_operation').operations

    if model_type in [op.id for op in data_operations]:
        return 'data_operation'
    else:
        return 'model'
