import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='Accuracy', train=False)

    def compute_metric(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        # print("--------------------计算精度时的output")
        # print(outputs)
        max_k = max(self.top_k)                                                                     # max_k = 1
        batch_size = labels.shape[0]                                                                # batch_size = 100
        _, pred = outputs.topk(max_k, 1, True, True)                                                # 输出最大值的索引和值,这里取的是索引
        pred = pred.t()                                                                             # 转置,从（2,1）变为（1,2）
        correct = pred.eq(labels.view(1, -1).expand_as(pred))                                       # b.expand_as(a)就是将b进行扩充，扩充到a的维度，需要说明的是a的低维度需要比b大，例如b的shape是31，如果a的shape是32不会出错，
        # correct = [[true] ,[true], [false]]这种                                                    eq()函数比较两向量是否,两个向量的维度必须一致，如果相等，对应维度上的数为1，若果不相等则对应位置上的元素为0.
        res = dict()
        for k in self.top_k:
            correct_k = correct[:k].view(-1).float().sum(0)    # view=reshape中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。view(-1)展成一排，.float()转为0和1，sum(0)
            res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        return res                                                                                  # res = {"top-1": 5 }，上一排就是将整数变成浮点数
