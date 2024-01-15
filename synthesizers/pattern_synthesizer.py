import random

import torch
from torchvision.transforms import transforms, functional
from torchvision.transforms import InterpolationMode
from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()                                            # 将PyTorch张量（Tensors）转换成PIL（Python Imaging Library）图像对象
transform_to_tensor = transforms.ToTensor()                                             # 将PIL图像对象或NumPy数组转换为PyTorch张量

# 可以理解为给图片加一个小块，位置和小块内容自己定（随机矩阵）
class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([                           # shape = (5,3)
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]])
    "Just some random 2D pattern. 图案"
    x_top = 0
    "X coordinate to put the backdoor into."
    y_top = 0
    "Y coordinate to put the backdoor into."
    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."
    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."
    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."
    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."
    def __init__(self, task: Task):
        super().__init__(task)
        # 对随机矩阵大小进行一次初始化
        # 在Python中，self 是一个特殊的关键字，通常用于表示对象自身。在类的方法中，self 用来引用类的实例变量和方法。
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)
    # 加小块，可以定义位置和内容
    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.params.input_shape)                               # 1.生成一个3*32*32的0张量input_shape=3*32*32
        full_image.fill_(self.mask_value)                                               # b.fill_(-10)就表示用-10填充b，也就是b的形状为3*32*32，值全为-1
        # print(f"full_image: {full_image}")
        # x_top x_bot分别表示x的起点和终点
        x_bot = x_top + pattern_tensor.shape[0]                                         # x_bot = 8 + 15 = 23 2.计算右下角的值
        # y_top y_bot分别表示y的起点和终点
        y_bot = y_top + pattern_tensor.shape[1]                                         # y_bot = 2 + 9 = 11
        # print(f"x_bot: {x_bot}, y_bot: {y_bot}")
        if x_bot >= self.params.input_shape[1] or y_bot >= self.params.input_shape[2]:  # 后门不可大于图片尺寸
            raise ValueError(f'Position of backdoor outside image limits: image: {self.params.input_shape}, but backdoor ends at ({x_bot}, {y_bot})')
        # 将触发器加在full_image上
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor                        # 3.full_image[:, 3:8, 23:26]，第一个值表示通道，目的是为了计算mask
        # print(f"full_image2: {full_image}")
        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)          # 判断full_image中的每个值，不等为1，想等为0，self.mask就是保存的这种值
        # print(f"mask: {self.mask}")   self.mask的形状为3*32*32
        self.pattern = self.task.normalize(full_image).to(self.params.device)           # 5.pattern进行标准化
        # 对于每个通道，它从每个像素的值中减去了一个均值（mean）值。通常，这个均值是 [0.485, 0.456, 0.406]，它们分别对应了红色、绿色和蓝色通道的均值。
        # 接着，它将每个通道的值除以一个标准差（std）值。通常，这个标准差是 [0.229, 0.224, 0.225]，它们分别对应了红色、绿色和蓝色通道的标准差。
        # 这个操作的目的是将图像的像素值缩放到一个标准范围，以便神经网络更容易进行训练。这是深度学习中常见的数据预处理步骤之一，有助于模型更快地收敛并提高性能。
        # 在深度学习中，这种数据预处理通常在训练数据和测试数据上都要进行，以确保模型在不同数据集上的表现一致。通常，这种操作会被包装成一个数据转换（transform）或数据加载器（dataloader）中的一部分，以便方便地应用于图像数据。
        # print(f"pattern3: { self.pattern}")

    # 在图片中加入pattern
    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()                                                                              # 在attack的portion上加入pattern
        # print("-----------------------")
        # print(f"pattern：{pattern}")
        # print(f"mask：{mask}")
        batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern
        return
    # 在标签中加入pattern
    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)                                                 # 在attack的portion上加入backdoor_label，这里backdoor_label是8
        return
    # 生成随机的pattern内容和位置
    def get_pattern(self):
        # pattern位置随机，内容大小随机
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])                                         # 1.[5,10]中随机选择一个整数
            # print(f"resize:{resize}")
            pattern = self.pattern_tensor                                                                               # 2.选择上面的5*3的pattern矩阵
            # if random.random() > 0.5:                                                                                   # 用于生成一个0到1的随机符点数
            #     pattern = functional.hflip(pattern)                                                                     # 将指定图像水平翻折
            image = transform_to_image(pattern)                                                                         # 将触发器转换为图片
            pattern = transform_to_tensor(functional.resize(image, resize, interpolation=InterpolationMode.NEAREST)).squeeze()    # 3.将输入的图像调整大小resize对标shape1、转换为张量，并去除不必要的维度，最终得到一个处理后的张量pattern
            # InterpolationMode.NEAREST表示使用最近邻插值方法。最近邻插值方法会根据目标像素的位置，在原图像中找到最接近的像素的值作为目标像素的值。
            # 这里需要注意的是。如果最近邻插值方法在为负数，取值则为>0的接近0的像素值
            # print(f"pattern2:{pattern}")
            # print(f"pattern2.shape:{pattern.shape}")                                                                    # 当resize=6时，pattern.shape=[10,6],当resize=9时，shape=[15,9]
            x = random.randint(0, self.params.input_shape[1] - pattern.shape[0] - 1)                                    # (0, 32-9) 4.确定左上角的xy值
            y = random.randint(0, self.params.input_shape[2] - pattern.shape[1] - 1)                                    # (0, 32-5)
            self.make_pattern(pattern, x, y)
        return self.pattern, self.mask
