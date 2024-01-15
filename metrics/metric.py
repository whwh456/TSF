import logging
from collections import defaultdict
from typing import Dict, Any

import numpy as np

logger = logging.getLogger('logger')


class Metric:
    name: str
    train: bool
    plottable: bool = True
    running_metric = None
    main_metric_name = None

    def __init__(self, name, train=False):
        self.train = train
        self.name = name
        # defaultdict(list),会构建一个默认value为list的字典，
        self.running_metric = defaultdict(list)                                 # defaultdict(list),会构建一个默认value为list的字典，

    def __repr__(self):                                                         # 该方法可以控制打印类时输出的内容
        metrics = self.get_value()
        text = [f'{key}: {val:.2f}' for key, val in metrics.items()]
        return f'{self.name}: ' + ','.join(text)

    # 这个错误其实是个提醒错误，在父类中定义一个方法，知道有这个方法，不知道如何实现或者不想实现，等有人继承他，就得帮他实现，不实现就报错，提醒你父类里有一个方法你没实现
    def compute_metric(self, outputs, labels) -> Dict[str, Any]:
        raise NotImplemented

    def accumulate_on_batch(self, outputs=None, labels=None):
        current_metrics = self.compute_metric(outputs, labels)                                                          # ACC=
        for key, value in current_metrics.items():
            self.running_metric[key].append(value)

    def get_value(self) -> Dict[str, np.ndarray]:
        metrics = dict()
        for key, value in self.running_metric.items():
            metrics[key] = np.mean(value)
        return metrics

    def get_main_metric_value(self):
        if not self.main_metric_name:
            raise ValueError(f'For metric {self.name} define attribute main_metric_name.')
        metrics = self.get_value()
        return metrics[self.main_metric_name]

    def reset_metric(self):
        # defaultdict(list),会构建一个默认value为list的字典，
        self.running_metric = defaultdict(list)

    def plot(self, tb_writer, step, tb_prefix=''):
        if tb_writer is not None and self.plottable:
            metrics = self.get_value()
            for key, value in metrics.items():
                tb_writer.add_scalar(tag=f'{tb_prefix}/{self.name}_{key}',scalar_value=value, global_step=step)
            tb_writer.flush()
        else:
            return False
