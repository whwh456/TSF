import random

import torch
from torchvision.transforms import transforms, functional
from torchvision.transforms import InterpolationMode
from synthesizers.synthesizer import Synthesizer
from tasks.task import Task
import numpy as np
# import albumentations
import cv2
# 可以理解为给图片加一个小块，位置和小块内容自己定（随机矩阵）
class BaikuaiSynthesizer(Synthesizer):
    '''A tensor of the `input.shape` filled with `mask_value` except backdoor.'''
    def __init__(self, task: Task):
        super().__init__(task)
        # 对随机矩阵大小进行一次初始化
        # 在Python中，self 是一个特殊的关键字，通常用于表示对象自身。在类的方法中，self 用来引用类的实例变量和方法。

    # 在图片中加入pattern
    def synthesize_inputs(self, batch, attack_portion=None):
        # batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern
        for i in range(attack_portion):  # x_train.shape[0]=50000
            # print("batch.inputs.shape", batch.inputs.shape) 
            # print("batch.inputs[i].shape", batch.inputs[i].shape)
            batch.inputs[i] = self.patching_train(batch.inputs[i], 0)
        return
    # 在标签中加入pattern
    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)                                                 # 在attack的portion上加入backdoor_label，这里backdoor_label是8
        return

    def patching_train(self, clean_sample, code):
        '''this code conducts a patching procedure with random white blocks or random noise block'''
        # np.random.randint(0,5)返回一个随机整型数，范围从低（包括）到高
        # attack = np.random.randint(0,5)

        # attack修补的类型
        attack = code
        pat_size_x = np.random.randint(2, 8)
        pat_size_y = np.random.randint(2, 8)  # x = 3 y = 4  白块为：3*3*4
        output = np.copy(clean_sample)
        if attack == 0:  # 被设置为一个大小为 (pat_size_x, pat_size_y, 3) 的全白块。
            # block = np.ones((3, pat_size_x, pat_size_y)) # 彩色图片用
            block = np.ones((pat_size_x, pat_size_y)) # 黑白图片用，例如mnist
        elif attack == 1:  # 被设置为一个大小为 (pat_size_x, pat_size_y, 3) 的随机值块。
            # block = np.random.rand(3, pat_size_x, pat_size_y) # 彩色图片
            block = np.random.rand(pat_size_x, pat_size_y) # 黑色图片用
        # elif attack == 2:
        #     return self.addnoise(output)
        elif attack == 3:
            return self.randshadow(output)
        margin = np.random.randint(0, 6)   # m = 3 rl = 2
        rand_loc = np.random.randint(0, 4)  # 根据 rand_loc 的不同取值，将修补块 block 放置在 output 中的不同位置。
        if rand_loc == 0:
            # output[:, margin:margin + pat_size_x, margin:margin + pat_size_y] = block  # 应用于彩色图片，upper left 4:4+6
            output[0, margin:margin + pat_size_x, margin:margin + pat_size_y] = block
        elif rand_loc == 1:
            # output[:, margin:margin + pat_size_x, 32 - margin - pat_size_y:32 - margin] = block
            output[0, margin:margin + pat_size_x, 28 - margin - pat_size_y:28 - margin] = block
        elif rand_loc == 2: # x = 3 y = 4  白块为：3*3*4   26:29, 3:7
            # output[:, 32 - margin - pat_size_x:32 - margin, margin:margin + pat_size_y] = block
            output[0, 28 - margin - pat_size_x:28 - margin, margin:margin + pat_size_y] = block
        elif rand_loc == 3:
            # output[:, 32 - margin - pat_size_x:32 - margin, 32 - margin - pat_size_y:32 - margin] = block  # right bottom
            output[0, 28 - margin - pat_size_x:28 - margin, 28 - margin - pat_size_y:28 - margin] = block

        output[output > 1] = 1
        return torch.from_numpy(output).float()

    # 加噪声
    # def addnoise(img):
    #     aug = albumentations.GaussNoise(p=1, mean=0, var_limit=(10, 1000))
    #     # 每个数乘以255，再转化为uint8，图片类型转换
    #     augmented = aug(image=(img * 255).astype(np.uint8))
    #     auged = augmented['image'] / 255
    #     return auged

    # 加阴影
    def randshadow(img):
        aug = albumentations.RandomShadow(p=1)
        test = (img * 255).astype(np.uint8)
        augmented = aug(image=cv2.resize(test, (32, 32)))
        auged = augmented['image'] / 255
        return auged
