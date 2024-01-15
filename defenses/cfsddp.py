import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from defenses.fedavg import FedAvg
import torch.nn.functional as F
logger = logging.getLogger('logger')

class CFSDDP(FedAvg):
    def aggr(self, hlpr, weight_accumulator, global_model, local_updates, user_id):
        # 进行预聚合操作，初始化一个字典用于存储预聚合全局模型
        for index, update in enumerate(local_updates):
            hlpr.task.accumulate_weights(weight_accumulator, update)
        global_update = dict()
        for name, sum_update in weight_accumulator.items():
            scale = self.params.fl_eta / self.params.fl_total_participants                               
            global_update[name] = scale * sum_update                                                             # 联邦平均更新
        # for name, sum_update in weight_accumulator.items():
        #     scale = self.params.fl_eta / self.params.fl_total_participants                                              # scale = 10/100
        #     global_update = scale * sum_update
        # global_update是张量
        # 计算相似度
        global_update_tensor = torch.cat([var.flatten() for var in global_update.values()])
        # 归一化，使得张量具有单位长度
        global_update_normalized = F.normalize(global_update_tensor, p=2, dim=0)
        
        # 将 global_update 和 local_updates 展平成一维张量
        scores = []
        for update in local_updates:
            update_tensor = torch.cat([var.flatten() for var in update.values()])
            update_normalized = F.normalize(update_tensor, p=2, dim=0)
            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(global_update_normalized, update_normalized, dim=0)
            scores.append(cosine_similarity)
        # 打印余弦相似度值
        print('余弦相似度值: ', [round(score.item(), 4) for score in scores])
        # 初始化一个列表用于存储检测到的恶意客户端的索引
        poison_client_index = []
        # 对于每一个本地更新
        for i in range(len(scores)):
            # 如果余弦相似度分数超过阈值
            if scores[i] > hlpr.params.threshold:
                poison_client_index.append(i)
        # 如果没有检测到恶意客户端，返回[-1]
        if poison_client_index:
            print("恶意更新终端为：", end = ' ')
            for index in poison_client_index:
                print(f'{user_id[index]}', end=' ')
        # 重新聚合接下来的更新
        weight_accumulator = hlpr.task.get_empty_accumulator()
        for index, update in enumerate(local_updates):
            if index in poison_client_index:
                continue
            hlpr.task.accumulate_weights(weight_accumulator, update)
    