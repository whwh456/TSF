import torch
from synthesizers.synthesizer import Synthesizer
import albumentations
import numpy as np
from tasks.task import Task
import cv2
# 随机噪声---加的是高斯噪声
class NoiseSynthesizer(Synthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that represents the physical object in the real scene.
    """

    def __init__(self, task: Task):
        super().__init__(task)

    # 在图片中加入pattern
    def synthesize_inputs(self, batch, attack_portion=None):
        # batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern
        for i in range(attack_portion):  # x_train.shape[0]=50000
            batch.inputs[i] = self.addnoise(batch.inputs[i])
        return

    # 在标签中加入pattern
    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)  # 在attack的portion上加入backdoor_label，这里backdoor_label是8
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
            block = np.ones((3, pat_size_x, pat_size_y))
        elif attack == 1:  # 被设置为一个大小为 (pat_size_x, pat_size_y, 3) 的随机值块。
            block = np.random.rand(3, pat_size_x, pat_size_y)
        elif attack == 2:
            return self.addnoise(output)
        elif attack == 3:
            return self.randshadow(output)

        margin = np.random.randint(0, 6)  # m = 3 rl = 2
        rand_loc = np.random.randint(0, 4)  # 根据 rand_loc 的不同取值，将修补块 block 放置在 output 中的不同位置。
        if rand_loc == 0:
            output[:, margin:margin + pat_size_x, margin:margin + pat_size_y] = block  # upper left 4:4+6
        elif rand_loc == 1:
            output[:, margin:margin + pat_size_x, 32 - margin - pat_size_y:32 - margin] = block
        elif rand_loc == 2:  # x = 3 y = 4  白块为：3*3*4   26:29, 3:7
            output[:, 32 - margin - pat_size_x:32 - margin, margin:margin + pat_size_y] = block
        elif rand_loc == 3:
            output[:, 32 - margin - pat_size_x:32 - margin,
            32 - margin - pat_size_y:32 - margin] = block  # right bottom

        output[output > 1] = 1
        return torch.from_numpy(output).float()
    # 加噪声
    def addnoise(self, img):
        # mean是高斯分布的均值，用于控制噪声的中心位置
        # var_limit这是高斯分布的方差（或标准差）的限制范围。方差用于控制高斯噪声的强度。在这里，方差的范围被设置为 (10, 1000)，意味着每次应用该变换时，方差将在10到1000之间随机选择，以产生不同程度的噪声。
        aug = albumentations.GaussNoise(p=1, mean=0, var_limit=(10, 1000))
        # 每个数乘以255，再转化为uint8，图片类型转换
        img = img.cpu().numpy()
        augmented = aug(image=(img * 255).astype(np.uint8))
        auged = augmented['image'] / 255
        return torch.from_numpy(auged).float()
    # 加阴影
    def randshadow(self, img):
        aug = albumentations.RandomShadow(p=1)
        img = img.numpy()
        test = (img * 255).astype(np.uint8)
        augmented = aug(image=cv2.resize(test, (32, 32)))
        auged = augmented['image'] / 255
        return torch.from_numpy(auged).float()