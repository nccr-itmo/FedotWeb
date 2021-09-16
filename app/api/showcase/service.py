import pickle
from typing import List, Optional

from app import storage
from app.api.data.service import get_dataset_metadata
from app.api.pipelines.service import pipeline_by_uid, get_pipeline_metadata
from init.init_cases import add_case_to_db
from .models import ShowcaseItem
from .showcase_utils import prepare_icon_path, showcase_item_from_db
from ..analytics.pipeline_analytics import get_metrics_for_pipeline


def showcase_item_by_uid(case_id: str) -> ShowcaseItem:
    item = showcase_item_from_db(case_id)
    return item


def showcase_full_item_by_uid(case_id: str) -> Optional[ShowcaseItem]:
    dumped_item = storage.db.cases.find_one({'case_id': case_id})
    if dumped_item is None:
        return None

    icon_path = prepare_icon_path(dumped_item)

    case_metadata = pickle.loads(dumped_item['metadata'])
    pipeline_id = dumped_item['pipeline_id']

    details = dumped_item['details']
    is_updated = False
    if not details:
        n_features, n_rows = get_dataset_metadata(case_metadata.dataset_name, 'train')
        n_models, n_levels = get_pipeline_metadata(pipeline_id)
        details = {
            'n_models': n_models,
            'n_levels': n_levels,
            'n_features': n_features,
            'n_rows': n_rows}

        case_id = dumped_item['case_id']
        pipeline, case = pipeline_by_uid(pipeline_id), showcase_item_by_uid(case_id)

        if pipeline is None:
            raise ValueError(f'Pipeline with id {pipeline_id} not exists.')

        metrics = get_metrics_for_pipeline(case, pipeline)
        for metric_id in metrics.keys():
            details[metric_id] = metrics[metric_id]
        is_updated = True

    item = ShowcaseItem(case_id=case_id,
                        title=dumped_item['title'],
                        icon_path=icon_path,
                        description=dumped_item['description'],
                        pipeline_id=pipeline_id,
                        metadata=case_metadata,
                        details=details)
    if is_updated:
        add_case_to_db(db=storage.db, case=item)
    return item


def all_showcase_items_ids() -> List[str]:
    items = storage.db.cases.find()
    ids = [item['case_id'] for item in items if 'case_id' in item]
    return ids
