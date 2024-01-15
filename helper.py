#工作包括：
# 1.判断是否是联邦学习
# 2.生成Synthesizer，为攻击模型准备。
# 3.attack包括了Synthesizer和一些计算loss的函数，可以进行多任务的操作。
# 4.tb_writer是tensorboard可视化结果的一个工具。


import importlib
import logging
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy
from shutil import copyfile
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from synthesizers.synthesizer import Synthesizer
from tasks.fl.fl_task import FederatedLearningTask
from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')

'''
    学习进度：
'''
class Helper:
    '''
    属性：
    方法：1.确定任务（返回一个对象实例） 2.确定合成器
    '''
    params: Params = None                                                      # 参数，当冒号后面是类时，就可以这么写
    task: Union[Task, FederatedLearningTask] = None                            # 任务，联合类型：Union[int, str] 表示既可以是 int，也可以是 str
    synthesizer: Synthesizer = None                                            # 合成器
    attack: Attack = None                                                      # 攻击
    tb_writer: SummaryWriter = None                                            # tensorboard的结果
    # 初始化任务
    def __init__(self, params):
        self.params = Params(**params)                                         # 关键字参数，数量可变
        # print(self.params)                                                   # 这里加载了配置文件的参数
        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        if self.params.random_seed is not None:                                # 不懂什么意思
            self.fix_random(self.params.random_seed)
        # 创建结果的文件夹
        self.make_folders()
        # 找到训练用的xx_task文件，用默认构造函数获取构造后的结果
        self.make_task()
        # if self.params.defense:
        #     self.make_defense()
        self.make_synthesizer()                                                 # 你选的什么synthesizer，就是什么合成后门
        self.attack = Attack(self.params, self.synthesizer)                     # 拿到攻击对象
        # neural cleanse 识别和减轻神经网络中的后门攻击的手段
        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc = True
        # if 'spectral_evasion' in self.params.loss_tasks:
        #     self.attack.fixed_model = deepcopy(self.task.model)
        self.best_loss = float('inf')
        # print(torch.cuda.is_available())
        # print(f"实验设备：{self.params.device}")

    # 1.通过cifar_fed中的task来确实采用什么方法训练，然后新建一个该文件下类的实例。
    def make_task(self):                                                        # 通过参数task来锁定模块中的类，并创建一个对象实例
        name_lower = self.params.task.lower()                                   # lower() 方法转换字符串中所有大写字符为小写。这里为name_lower = cifarfed
        name_cap = self.params.task                                             # name_cap = CifarFed
        if self.params.fl:                                                      # 如果是FL任务
            module_name = f'tasks.fl.{name_lower}_task'                         # module_name = tasks.fl.cifarfed_task
            path = f'tasks/fl/{name_lower}_task.py'                             # path = tasks/fl/cifarfed_task.py
        else:
            module_name = f'tasks.{name_lower}_task'                            # module_name = tasks.mnist_task，表示模块名字
            path = f'tasks/{name_lower}_task.py'                                # path = tasks/mnist_task.py
        try:

            task_module = importlib.import_module(module_name)                  # task_module = （动态地获取另一个py文件中定义好的变量/方法）tasks.fl.cifarfed_task
            task_class = getattr(task_module, f'{name_cap}Task')                # task_class = （从task_module中获取MNISTTask的属性值）是个CifarFedTask类
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should be defined as a class {name_cap}Task in {path}')
        self.task = task_class(self.params)                                     # self.task = CifarFedTask(参数)
    
    def make_defense(self):
        name_lower = self.params.defense.lower()
        name_cap = self.params.defense
        module_name = f'defenses.{name_lower}'
        try:
            defense_module = importlib.import_module(module_name)
            defense_class = getattr(defense_module, f'{name_cap}')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your defense: {self.params.defense} should '
                                      f'be one of the follow: FLAME, Deepsight, \
                                        Foolsgold, FLDetector, RFLBAT, FedAvg')
        self.defense = defense_class(self.params)
    
    # 2.通过cifar_fed中的synthesizer来确实采用什么攻击方法，同时创建一个该攻击方法的实例
    def make_synthesizer(self):                                                 # 根据参数锁定合成器对象，并返回实例
        name_lower = self.params.synthesizer.lower()                            # name_lower = pattern
        name_cap = self.params.synthesizer                                      # name_cap = Pattern
        module_name = f'synthesizers.{name_lower}_synthesizer'                  # module_name = synthesizers.pattern_synthesizer
        try:
            synthesizer_module = importlib.import_module(module_name)           # synthesizer_module = synthesizers.pattern_synthesizer
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')  # task_class = 是一个对象PatternSynthesizer
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)                                # 返回一个PatternSynthesizer对象实例

    # 1.根据params的log是否为True判断是否要在params_folder_path创建文件夹；2.文件夹中还包含run.html，一些画图内容包含在里面；
    # 创建日志信息   3.还会根据tb是否为True来判断是否要使用Tensorboard作图。
    def make_folders(self):
        log = create_logger()                                                   # 创建日志环境
        if self.params.log:
            try:
                os.mkdir(self.params.folder_path)                               # saved_models/model_  os.mkdir() 方法用于以数字权限模式创建目录（单级目录），默认的模式为 0777 (八进制)。
            except FileExistsError:
                log.info('Folder already exists')
            # with open('saved_models/runs.html', 'a') as f:
            #     f.writelines([f'<div>'
            #                   f'    <a href="https://github.com/ebagdasa/backdoors/tree/{self.params.commit}">GitHub</a>, '
            #                     f'  <span> '
            #                     f'       <a href="http://gpu/{self.params.folder_path}">{self.params.name}_{self.params.current_time}</a>'
            #                   f'    </span>'
            #                   f'</div>'])
            fh = logging.FileHandler(filename=f'{self.params.folder_path}/log.txt')                                     # 将日志发送到磁盘，默认无限增长
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')                       # 输出格式
            fh.setFormatter(formatter)
            log.addHandler(fh)
            log.warning(f'Logging to: {self.params.folder_path}')
            log.error(f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/{self.params.commit}">https://github.com/ebagdasa/backdoors/tree/{self.params.commit}</a>')
            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)                                                                               # yaml.dump()函数，就是将yaml文件一次性全部写入你创建的文件。
        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')                                                      # 将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用。
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)                                                              # 把模型参数字典写入日志
    
    def save_update(self, model=None, userID = 0):
        folderpath = '{0}/saved_updates'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)    
    
    # 3.保存模型
    def save_model(self, model=None, epoch=0, val_loss=0):
        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")                                      # saved_models/model_{self.task}_{self.current_time}_{self.name}
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
            saved_dict = {'state_dict': model.state_dict(),
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)                                                         # 保存断点
            if epoch in self.params.save_on_epochs:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss
    # 保存断点
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)                                                                                     # 保存模型
        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()                                                                                      # 关闭文件

    def plot(self, x, y, name):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=x)                                          # add_scalar()函数的目的是添加一个标量数据（scalardata）到summary中
            self.flush_writer()                                                                                         # 关闭文件
        else:
            return False

    def report_training_losses_scales(self, batch_id, epoch):                                                           # 本地训练时打印出来了
        if not self.params.report_train_loss or batch_id % self.params.log_interval != 0:                               # log_interval记录时间间隔
            return
        total_batches = len(self.task.train_loader)
        losses = [f'{x}: {np.mean(y):.3f}' for x, y in self.params.running_losses.items()]
        scales = [f'{x}: {np.mean(y):.3f}' for x, y in self.params.running_scales.items()]
        logger.info(f'Epoch: {epoch:3d}. Batch: {batch_id:5d}/{total_batches}.  Losses: {losses}. Scales: {scales}')
        for name, values in self.params.running_losses.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values), f'Train/Loss_{name}')
        for name, values in self.params.running_scales.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),f'Train/Scale_{name}')
        self.params.running_losses = defaultdict(list)                                                                  # 这里表示清空数据吧
        self.params.running_scales = defaultdict(list)

    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn
        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)
        return True
