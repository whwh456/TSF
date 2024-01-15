import math
from typing import List, Any, Dict
import torch
import logging
import os
from utils.parameters import Params

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FedAvg:
    params: Params
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def __init__(self, params: Params) -> None:
        self.params = params

    # FedAvg aggregation
    def aggr(self, weight_accumulator, _):
        raise NotImplemented