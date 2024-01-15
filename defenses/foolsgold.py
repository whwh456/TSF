import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')

class Foolsgold(FedAvg):
    # 用于保存训练历史
    def save_history(self, userID = 0):
        # 定义保存历史的文件夹路径
        folderpath = '{0}/foolsgold'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        # 定义历史文件的名称
        history_name = '{0}/history_{1}.pth'.format(folderpath, userID)
        # 定义更新文件的名称
        update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, userID)
        # 从更新文件中加载模型
        model = torch.load(update_name)
        # 如果历史文件已经存在
        if os.path.exists(history_name):
            # 从历史文件中加载参数
            loaded_params = torch.load(history_name)
            history = dict()
            # 遍历加载的参数，将其与当前模型相加，并保存到历史中
            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                history[name] = data + model[name]
            # 保存更新后的历史
            torch.save(history, history_name)
        else:
            # 如果历史文件不存在，则直接保存当前模型到历史文件
            torch.save(model, history_name)
    # 聚合参数
    def aggr(self, weight_accumulator, _):

        for i in range(self.params.fl_total_participants):
            self.save_history(userID = i)

        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        epsilon = 1e-5
        folderpath = '{0}/foolsgold'.format(self.params.folder_path)
        # Load params
        his = []
        for i in range(self.params.fl_total_participants):
            history_name = '{0}/history_{1}.pth'.format(folderpath, i)
            his_i_params = torch.load(history_name)
            for name, data in his_i_params.items():
                # his_i = np.append(his_i, ((data.cpu().numpy()).flatten()))
                if layer_name in name:
                    his = np.append(his, (data.cpu().numpy()).flatten())
        his = np.reshape(his, (self.params.fl_total_participants, -1))
        logger.info("FoolsGold: Finish loading history updates")
        cs = smp.cosine_similarity(his) - np.eye(self.params.fl_total_participants)
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.params.fl_total_participants):
            for j in range(self.params.fl_total_participants):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        logger.info("FoolsGold: Calculate max similarities")
        # Pardoning
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
    
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        
        # Federated SGD iteration
        logger.info(f"FoolsGold: Accumulation with lr {wv}")
        for i in range(self.params.fl_total_participants):
            update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            update_params = torch.load(update_name)
            for name, data in update_params.items():
                if self.check_ignored_weights(name):
                    continue
                weight_accumulator[name].add_((wv[i]*data).to(self.params.device))
        return weight_accumulator