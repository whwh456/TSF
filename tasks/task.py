import logging
from typing import List

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms


from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from tasks.batch import Batch
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    scheduler: CosineAnnealingLR = None
    metrics: List[Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()
    # 初始化任务：包括加载数据、模型、优化器、损失函数、评估指标、样本尺寸，将模型加到设备上
    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)
        self.optimizer = self.make_optimizer()
        # 返回交叉熵损失
        self.criterion = self.make_criterion()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented
    # 交叉熵
    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.
        We use reduction `none` to support gradient shaping defense.
        我们使用reduction' none '来支持梯度整形防御。
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')                    # 最常用的参数为 reduction(str, optional) ，可设置其值为 mean, sum, none ，默认为 none。该参数主要影响多个样本输入时，损失的综合方法。mean表示损失为多个样本的平均值，sum表示损失的和，none表示不综合
    # 选择梯度下降算法
    def make_optimizer(self, model=None) -> Optimizer:                                  # 选择优化器
        if model is None:                                                               # 判断模型为空
            model = self.model
        if self.params.optimizer == 'SGD':                                              # 随机梯度下降
            optimizer = optim.SGD(model.parameters(), lr=self.params.lr, weight_decay=self.params.decay, momentum=self.params.momentum)
        elif self.params.optimizer == 'Adam':                                           # 动量和自适应学习率优化下降，SGD升级版
            optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')
        return optimizer
    # 创建一个学习率调整器
    def make_scheduler(self) -> None:
        if self.params.scheduler:
            # 表示要对哪个优化器的学习率进行调度。T_max 是余弦退火的周期，通常设置为训练的总周期数（epochs），这是一个参数。
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.params.epochs)                                # 创建一个余弦退火学习率调度器
        '''
        以下是一些常见的学习率调度策略：
            固定学习率：在整个训练过程中保持学习率不变。这是最简单的策略之一，适用于小型数据集和简单的模型。
            
            学习率衰减：在每个一定的训练周期或者当某个条件满足时，将学习率乘以一个小于1的因子。常见的衰减方法包括：
                StepLR：在固定的周期上衰减学习率。
                ReduceLROnPlateau：基于验证集性能来自适应地降低学习率。
            余弦退火（Cosine Annealing）：学习率在余弦函数的形状下进行周期性的变化。这种方法可以帮助模型跳出局部最小值，并在训练过程中逐渐降低学习率。
            
            三角形学习率策略（Triangular Learning Rate）：学习率在一个上升阶段和一个下降阶段之间交替变化，用于加速训练。

            一周期学习率策略（One-Cycle Learning Rate）：将学习率设置为一个周期内的变化模式，包括快速增加和减小。这有助于快速收敛。

            多项式学习率策略：学习率根据多项式函数进行调整，可以是线性、二次或更高次数的多项式。

            自适应学习率方法：一些优化算法，如Adam、Adagrad和RMSprop，自身包含学习率调整机制，因此不需要显式的学习率调度策略。

            差分学习率策略：用于保护数据隐私的差分隐私学习率策略，通过向梯度添加噪音来提高隐私性。

            超参数优化：使用自动超参数优化方法，如贝叶斯优化或网格搜索，来搜索最佳的学习率。
        '''
    # 继续训练模型
    def resume_model(self):
        if self.params.resume_model:
            logger.info(f'Resuming training from {self.params.resume_model}')
            loaded_params = torch.load(f"saved_models/" f"{self.params.resume_model}", map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            self.params.lr = loaded_params.get('lr', self.params.lr)
            logger.warning(f"Loaded parameters from saved model: LR is"
                           f" {self.params.lr} and current epoch is"
                           f" {self.params.start_epoch}")
    # 设置样本的尺寸：input_shape32*32
    def set_input_shape(self):
        inp = self.train_dataset[0][0]                                                                      # 表示取第1个样本的图片数据
        self.params.input_shape = inp.shape

    #把数据加载到GPU或者CPU上
    def get_batch(self, batch_id, data) -> Batch:
        inputs, labels = data # 适用于mnist和cifar10
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def accumulate_metrics(self, outputs, labels):
        for metric in self.metrics:
            metric.accumulate_on_batch(outputs, labels)                                                                 # 这里是两个评估数值,一个ACC一个loss

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def report_metrics(self, step, prefix='',tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:                                                                                     # 这里是两个评估数值,一个ACC一个loss
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        logger.warning(f'{prefix} {step:3d}.    {" | ".join(metric_text)}')                                             # Backdoor {str(backdoor):5s}. Epoch:{step:3d}.
        return self.metrics[0].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res
