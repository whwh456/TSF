import logging
from typing import Dict
import pdb
from scipy import signal
import torch
from copy import deepcopy
import numpy as np
from models.model import Model
from models.nc_model import NCModel
from synthesizers.synthesizer import Synthesizer
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params
from scipy.fftpack import dct, idct
# from phe import paillier
import cv2

logger = logging.getLogger('logger')


class Attack:
    params: Params
    synthesizer: Synthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()                                                                              # 创建了一个空列表，这个列表的主要目的是用来存储或记录损失值（loss）的历史数据。
    # fixed_model: Model

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer

        # NC hyper params
        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)

    # 计算视而不见的的损失
    def compute_blind_loss(self, model, criterion, batch, attack):
        """
        :param model: 模型
        :param criterion: 损失函数
        :param batch: 批次
        :param attack: Do not attack at all. Ignore all the parameters  完全不攻击
        :return:
        """
        batch = batch.clip(self.params.clip_batch)                                                      # 将batch的值限制在clip_batch范围内，这里不裁剪。这里的clip_batch为None
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']                            # loss_tasks=['backdoor', 'normal']。如果是攻击，就loss_tasks = 同时训练后门任务和正常任务
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack)                         # 后门批次攻击，还不是很懂,synthesizer.py
        # 自己的客户端防御
        if self.params.spectre_filter:
            # 对每个图片进行DCT转换，并进行过滤操作
            batch_back = self.local_spectre(batch_back)
        scale = dict()
        if 'neural_cleanse' in loss_tasks:                                                              # 先不管，没有
            self.neural_cleanse_part1(model, batch, batch_back)
        if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold or len(self.loss_hist) < 1000): # 先不管，没有
            loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(loss_tasks, self, model, criterion, batch, batch_back, compute_grad=False)
        elif self.params.loss_balance == 'MGDA':                                                        # loss_balance多任务学习时的损失平衡
            loss_values, grads = compute_all_losses_and_grads(loss_tasks, self, model, criterion, batch, batch_back, compute_grad=True) # 计算主任务和后门任务的损失值
            if len(loss_tasks) > 1:                                                                                     # 损失和梯度都是字典
                scale = MGDASolver.get_scales(grads, loss_values, self.params.mgda_normalize, loss_tasks)               # 获取比例
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(loss_tasks, self, model, criterion, batch, batch_back, compute_grad=False)
            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')
        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        self.loss_hist.append(loss_values['normal'].item())                                                             # 将正常损失值添加到损失历史中
        self.loss_hist = self.loss_hist[-1000:]                                                                         # 只保留最近1000个损失历史
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)                                                  # 缩放损失
        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):                                                                             # 将每个任务的损失值和权重添加到running_losses和running_scales中
            self.params.running_losses[t].append(loss_values[t].item())                                                 # running_losses训练时的损失，包括主任务和后门
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]                                                                  # 如果是第一个任务，直接将损失值乘以权重
            else:
                blind_loss += scale[t] * loss_values[t]                                                                 # 如果不是第一个任务，将之前的blind_loss加上当前任务的损失值乘以权重
        self.params.running_losses['total'].append(blind_loss.item())                                                   # 将总的blind_loss添加到running_losses中
        if blind_loss.item() == 0:
            raise ValueError("The loss is zero, which will cause gradient vanishing. Please modify the loss function.")
        return blind_loss

    def neural_cleanse_part1(self, model, batch, batch_back):
        self.nc_model.zero_grad()
        model.zero_grad()

        self.nc_model.switch_grads(True)
        model.switch_grads(False)
        output = model(self.nc_model(batch.inputs))
        nc_tasks = ['neural_cleanse_part1', 'mask_norm']

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_values, grads = compute_all_losses_and_grads(nc_tasks,
                                                          self, model,
                                                          criterion, batch,
                                                          batch_back,
                                                          compute_grad=False
                                                          )
        # Using NC paper params
        logger.info(loss_values)
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        self.nc_model.switch_grads(False)
        model.switch_grads(True)
    # 目的是调整本地权重更新值
    def fl_scale_update(self, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(self.params.fl_weight_scale)                         # 把x和y点对点相乘，保存在value中。value = value*100
        if self.params.Homomorphic_encryption:
            public_key, private_key = paillier.generate_paillier_key_pair()
            for key, value in local_update.items():
                local_update[key] = paillier.encrypt(value, public_key)

    def local_spectre(self, batch):  # 本地频谱过滤
        # 先设计低通滤波器
        b0, a0 = self.lowpassFilter()
        # 转换图像，batch有batch.inputs[0]、batch.labels[0]
        filter_batch = batch.clone()
        # 将图像转换为灰度图像，并利用数字低通滤波器对图片进行滤波
        for i in range(len(batch.inputs)):
            # 转为黑白图片
            # print(f"i = {i}, len(batch.inputs) = {len(batch.inputs)}, params.batch_size={self.params.batch_size}")
            gray_image = torch.mean(batch.inputs[i], dim=0)
            # 将NumPy数组复制以确保没有负步幅
            filter_img = signal.filtfilt(b0, a0, gray_image)
            filter_batch.inputs[i] = torch.from_numpy(np.copy(filter_img))
        return filter_batch

    # 低通滤波器设计
    def lowpassFilter(self):
        # 设计巴特沃斯低通滤波器参数：先归一化，然后转换到模拟滤波器指标
        wp = 2 * self.params.f_pass / self.params.f_sample
        ws = 2 * self.params.f_stop / self.params.f_sample
        Wp = 2 * self.params.f_sample * np.tan(wp / 2)
        Ws = 2 * self.params.f_sample * np.tan(ws / 2)
        # 求滤波器的阶数和3db系数，为true表示模拟滤波器
        N, Wn = signal.buttord(Wp, Ws, self.params.Ap, self.params.As, analog=True)
        # 求低通滤波器的系数
        b, a = signal.butter(N, Wn, btype='low', analog=True)
        # 双线性转换为数字滤波器
        b0, a0 = signal.bilinear(b, a, fs=self.params.f_sample)
        return b0, a0

    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

