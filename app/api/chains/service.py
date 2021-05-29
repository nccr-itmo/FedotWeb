import warnings
from typing import Tuple

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import validate
from fedot.core.chains.node import PrimaryNode, SecondaryNode

from app import mongo


def chain_first():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def chain_mock():
    #      XG
    #   |      \
    #  XG      KNN
    #  | \      |  \
    # LR XG   LR   LDA
    #    |  \
    #   KNN  LDA
    new_node = SecondaryNode('xgboost')
    for model_type in ('knn', 'pca'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain = chain_first()
    chain.update_subtree(chain.root_node.nodes_from[0].nodes_from[1], new_node)
    return chain


def chain_by_uid(uid: str) -> Chain:
    chain = Chain()
    resp = mongo.db.chains.find_one({'uid': uid})
    chain.load(resp)
    return chain


def validate_chain(chain: Chain) -> Tuple[bool, str]:
    try:
        validate(chain)
        return True, 'Correct chain'
    except ValueError as ex:
        return False, str(ex)


def create_chain(uid: str, chain: Chain):
    # TODO search chain with same structure and data in database
    existing_uid = 'test_chain'
    is_new = uid != existing_uid

    if is_new:
        # TODO save chain to database
        warnings.warn('Cannot create new chain')
        uid = 'new_uid'

    return uid, is_new
