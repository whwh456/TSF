import sys
from dataclasses import dataclass
import torch


@dataclass
class Batch:
    batch_id: int
    inputs: torch.Tensor
    labels: torch.Tensor

    # For PIPA experiment we use this field to store identity label.
    aux: torch.Tensor = None

    def __post_init__(self):
        self.batch_size = self.inputs.shape[0]                              # batch_size = 64
    # 把标签和数据加在到CPU或者GPU上
    def to(self, device):
        # 把标签和数据加在到CPU或者GPU上
        inputs = self.inputs.to(device)
        labels = self.labels.to(device)
        # print("-------获取GPU----------")
        # print(device)
        # sys.exit(1)
        if self.aux is not None:
            aux = self.aux.to(device)
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)
    # 复制数据和标签
    def clone(self):
        inputs = self.inputs.clone()
        labels = self.labels.clone()
        if self.aux is not None:
            aux = self.aux.clone()
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)
    # batch_size想要保留的数据的批次大小。这个参数的目的是限制数据，只保留前 batch_size 个样本，而删除其余的样本。
    def clip(self, batch_size):
        if batch_size is None:                                              # batch_size = none
            return self
        inputs = self.inputs[:batch_size]
        labels = self.labels[:batch_size]
        if self.aux is None:
            aux = None
        else:
            aux = self.aux[:batch_size]
        # 方法创建一个新的 Batch 对象，将 batch_id 设置为与原始对象相同，但数据限制为前 batch_size 个样本的 inputs、labels 和 aux 数据。
        return Batch(self.batch_id, inputs, labels, aux)