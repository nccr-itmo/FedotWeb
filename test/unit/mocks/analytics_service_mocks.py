from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MockIndividualGraph:
    depth: int


@dataclass
class MockIndividual:
    fitness: float
    graph: MockIndividualGraph


@dataclass
class MockOptHistory:
    individuals: List[List[MockIndividual]]


@dataclass
class MockMetaData:
    dataset_name: str = ''
    task_name: str = ''


@dataclass
class MockShowcaseItem:
    metadata: MockMetaData = MockMetaData()


@dataclass
class MockInputData:
    pass


@dataclass
class MockOutputData:
    predict: np.ndarray = None


@dataclass
class MockPipeline:
    is_fitted: bool = False
    should_return_baseline: bool = False  # non-existing field

    def fit(self, *args, **kwargs):
        self.is_fitted = True

    def predict(self, *args, **kwargs):
        if not self.is_fitted:
            raise ValueError()
        return MockOutputData()
