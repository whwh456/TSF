import math
import random
import sys
from copy import deepcopy
from typing import List, Any, Dict

from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
import torch
import logging
from torch.nn import Module
# from phe import paillier

from tasks.task import Task
logger = logging.getLogger('logger')

# 本函数从make_task()中进入
# 工作：
# 1.创建一个训练好的残差网络/恢复一个残差网络。
# 2.判断使用gpu还是cpu进行训练。
# 3.选择使用的评价标准（这里使用的是cross entrophy）
class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']                                           #['tracked', 'running']
    adversaries: List[int] = None
    # 初始化任务：包括加载数据、本地模型和全局模型，敌手数量、损失函数、评价指标，样本尺寸大小
    def init_task(self):
        self.load_data()                                                                # load_data对应的是cifarfed_task.py
        self.model = self.build_model()                                                 # build_model就只是创建一个18层的残差网络
        self.resume_model()                                                             # resume_model是当有训练过的模型后，则会使用之前的模型
        self.model = self.model.to(self.params.device)                                  # 使用cpu或者gpu训练，这里主要是使用cpu
        self.local_model = self.build_model().to(self.params.device)                    # 加载本地模型到dev上
        self.criterion = self.make_criterion()                                          # 直接用entrophy了，
        self.adversaries = self.sample_adversaries()                                    # 这个是选择攻击的客户端
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]               # 评价矩阵
        self.set_input_shape()                                                          # 表示取第一个样本的形状
        return

    # 在每一轮中：100个随机选择10个客户端，这里需要加载攻击者，返回参与者集合
    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(range(self.params.fl_total_participants), self.params.fl_no_models)                 # sampled_ids = 从100中随机选10个
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):                                                                     # 对于每个随机选中的client
            train_loader = self.fl_train_loaders[user_id]                                                               # train_loader = [客户id]
            compromised = self.check_user_compromised(epoch, pos, user_id)                                              # 判断是否是攻击者
            user = FLUser(user_id, compromised=compromised, train_loader=train_loader)
            sampled_users.append(user)
        return sampled_users
    # 标记攻击者，处理单轮攻击的攻击者id
    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.
        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack is not False:                                                             # 是单轮攻击，就需要设置fl_number_of_adversaries且小于10
            if epoch == self.params.fl_single_epoch_attack:                                                             # 当前批次为攻击批次，针对单次攻击的时候
                if pos < self.params.fl_number_of_adversaries:                                                          # 如果pos=[0-9]小于 攻击者数。
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised user: {user_id}.')
        else:                                                                                                           # 多轮攻击
            compromised = user_id in self.adversaries
        return compromised
    # 输出攻击者的信息，针对单轮攻击和多轮攻击两种系列来说明，会返回多轮攻击的id
    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        # 对应cifar_fed.yaml第45行，
        if self.params.fl_number_of_adversaries == 0:                                   # 无攻击情况
            logger.warning(f'Running vanilla FL, no attack.')                           # vanilla 寻常的，没有新意的
        elif self.params.fl_single_epoch_attack is False:                                # 不是单轮攻击
            adversaries_ids = random.sample(range(self.params.fl_total_participants), self.params.fl_number_of_adversaries) # 从100个随机选择几个攻击者并显示选择的攻击者
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.\ncompromised proportion '
                           f'= {self.params.fl_number_of_adversaries/self.params.fl_total_participants}')
        else:                                                                           # 单次攻击
            logger.warning(f'Attack only on epoch: '                                    # Attack only on epoch: 44 with 12 攻击者
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids

    # 获取模型优化器
    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)
        optimizer = self.make_optimizer(local_model)
        return local_model, optimizer

    # 用全局模型来更新本地参数，无返回值
    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    # 计算本地上传更新（这里可以选择用不用同态加密），返回本地更新local_update，也是模型参数（字典类型）
    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = self.get_empty_accumulator()
        for name, data in local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])
        if self.params.Homomorphic_encryption:
            public_key, private_key = paillier.generate_paillier_keypair()
            # print(f"公钥：{public_key.n}， 私钥：{private_key}")
            for key, value in local_update.items():
                local_update[key] = public_key.encrypt(value)
        return local_update

    # 计算L2范数update_norm（求平方和再开方）==>得到更新范数，可以用DP来裁剪
    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            # 如果本地更新中某个权重是和要忽略的权重匹配则跳过当前循环，进入下一轮循环
            if self.check_ignored_weights(name):
                continue
            #  计算更新值的每个元素的平方，计算平方后的值的总和，.item() 方法将总和转换为Python标量值
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        # 计算 squared_sum 的平方根，这将给出更新的范数（L2范数）。
        update_norm = math.sqrt(squared_sum)

        '''
        计算L2范数（也称为Euclidean范数或欧几里德范数）在深度学习和机器学习中有着重要的作用，主要用于以下几个方面：

            权重衰减（Weight Decay）：L2范数常用于正则化模型。正则化的目的是降低模型的复杂性，防止过拟合。通过将L2范数添加到损失函数中，
            可以鼓励模型权重保持较小的值，从而降低模型的复杂性。这有助于提高模型的泛化能力，使其在未见过的数据上表现更好。

            梯度裁剪（Gradient Clipping）：在训练神经网络时，梯度裁剪是一种控制梯度爆炸问题的技术。L2范数可以用来测量梯度的大小，如果梯度
            的L2范数超过了一个阈值，可以对梯度进行缩放，以防止梯度爆炸。

            模型监测与调试：计算权重更新的L2范数可以用来监测模型训练的进程。如果L2范数在训练过程中显著变化，这可能表明模型权重发生了大的变化，
            这可以作为模型性能不稳定的标志。

            优化器的调整：某些优化算法（如Momentum、Adam等）可能依赖于梯度信息。了解权重更新的L2范数可以帮助选择适当的优化器和超参数。

            如果不计算L2范数会怎么样呢？在某些情况下，你可能不需要计算L2范数，特别是当你不关心权重的大小或不使用正则化时。然而，通常情况下，
            使用L2范数作为正则化项可以帮助控制模型的复杂性，提高模型的泛化能力，并帮助解决训练过程中的一些数值稳定性问题。因此，是否计算L2范数
            通常取决于你的模型和训练任务的需求。
        '''
        return update_norm

    # 忽略的权重，意思是指有些权重要忽略掉，上面有个忽略权重列表，下面是匹配某个name是否在列表中
    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False

    # 得到一个空的权重类型weight_accumulator，数据类型为字典型
    def get_empty_accumulator(self):
        # 权重累积：是个字典
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    # 差分隐私裁剪本地更新
    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_diff_privacy and update_norm > self.params.fl_dp_clip:                                        # 如果加了DP，且更新范数>dp裁剪范数
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)
            ''' 
            在联邦学习中本地应用差分隐私时，梯度裁剪是一种常见的技术，它有助于保护个体参与者的隐私，同时允许模型在联邦学习过程中有效地进行训练。
            以下是为什么在联邦学习中进行梯度裁剪的一些原因：

                ①差分隐私保护：联邦学习通常涉及多个参与者，每个参与者拥有自己的本地数据。在聚合全局模型之前，本地参与者会计算其对模型的梯度，
                并将它们汇总到中央服务器以更新全局模型。在这个过程中，梯度可能包含敏感信息，如个体数据的细节。梯度裁剪可以限制梯度的范数，
                以减小梯度中包含的隐私信息泄露的风险，从而增强差分隐私的保护。

                ②减小梯度爆炸问题：在深度学习中，梯度可能会变得非常大，导致数值不稳定性和训练问题。梯度爆炸问题可能会在联邦学习中变得更加严重，
                因为梯度是从多个参与者汇总而来的。梯度裁剪可以限制梯度的大小，以防止梯度爆炸，有助于保持数值稳定性。

                ③控制模型更新的大小：梯度裁剪允许对模型更新的大小进行控制。这对于保持联邦学习中的模型同步和稳定性非常重要。如果某个本地参与者
                的更新过大，它可能会对全局模型造成不稳定的影响，甚至导致模型的性能下降。通过裁剪梯度，可以确保所有本地更新的大小都在可接受的范
                围内，有助于维持模型的整体稳定性。

                ④降低攻击风险：在联邦学习中，存在一些可能会尝试通过分析模型的更新来获取敏感信息的威胁，这被称为反推攻击。梯度裁剪可以降低攻击
                者获取有关模型和本地数据的信息的风险，因为裁剪后的梯度信息更难以利用。

                总之，梯度裁剪在联邦学习中是一项关键的隐私和稳定性技术。它帮助平衡了模型的训练效率、模型性能和差分隐私的要求，从而促进了联邦学
                习的成功应用。
            
            梯度爆炸的结果：
                梯度爆炸是深度学习中一种常见的问题，它会导致模型训练变得不稳定，并且可能会对模型的收敛性能和泛化性能产生负面影响。梯度爆炸通常
                在反向传播期间发生，即梯度计算的过程中，导致梯度值变得非常大。以下是梯度爆炸可能导致的结果：

                ①数值不稳定：梯度爆炸会导致梯度值变得非常大，可能远远超过计算机的浮点数表示范围。这会导致数值不稳定性，使权重和梯度的值变得异常。

                ②训练失败：当梯度爆炸发生时，权重更新可能会变得非常大，从而导致权重值迅速增加，使模型变得不稳定。在极端情况下，权重可能会趋向
                无穷大，这通常导致训练过程无法继续进行，或者损失函数出现NaN（不是数字）的情况，导致训练失败。

                ③梯度下降不收敛：梯度爆炸可以导致梯度下降算法失效，使模型无法收敛到合适的解决方案。训练过程可能会变得非常缓慢，或者根本无法在
                有限的时间内达到收敛。

                ④泛化性能下降：即使模型成功训练完成，梯度爆炸也可能导致模型在未见过的数据上的性能下降。这是因为模型过度拟合了训练数据，无法泛化到其他数据。

                为了解决梯度爆炸的问题，通常需要采取一些技术措施，如梯度裁剪、使用合适的权重初始化策略、使用更适合的激活函数（如ReLU代替Sigmoid或Tanh）
                等。梯度裁剪是一种常见的方法，可以有效地防止梯度爆炸。此外，选择合适的优化算法和学习率也对梯度爆炸的处理至关重要。
            
            一些常见的梯度裁剪技术以及它们的优缺点：
                全局梯度裁剪（Global Gradient Clipping）：
                    优点：简单而易于实现，适用于多种深度学习框架。可以确保整体梯度的范数不超过某个预定的阈值。
                    缺点：不考虑每个参数的敏感性，可能会导致信息泄露。对于不同的参数可能需要不同的裁剪阈值，但全局裁剪只使用一个阈值。
                自适应梯度裁剪（Adaptive Gradient Clipping）：
                    优点：考虑了每个参数的敏感性，因此可以更好地保护隐私。 可以根据参数的敏感性自动调整裁剪阈值。
                    缺点：实现相对复杂，可能需要额外的计算开销。需要额外的隐私分析来估计参数的敏感性。
                局部梯度裁剪（Local Gradient Clipping）：
                    优点：每个参与者可以在本地对其自己的梯度进行裁剪，无需将梯度传输到中央服务器，降低了通信开销。参与者可以根据其本地数据的特性自适应地裁剪梯度。
                    缺点：需要参与者信任他们自己的裁剪实现，可能存在安全性问题。可能需要更多的参与者协作和协调，以确保差分隐私的保护。
                分布式梯度裁剪（Distributed Gradient Clipping）：
                    优点：可以在多个参与者之间共享敏感性信息，以更好地保护隐私。可以根据不同的参与者和参数进行动态的敏感性分析和裁剪。
                    缺点：需要复杂的通信和协调机制，增加了系统的复杂性和开销。可能需要参与者共享一些敏感信息，可能引发隐私担忧。
                总的来说，梯度裁剪技术是联邦学习中保护差分隐私的重要手段之一。选择哪种技术取决于具体的应用场景和隐私需求。全局梯度裁剪简单，
                但可能不足以提供充分的隐私保护；自适应梯度裁剪更精细，但计算复杂；局部和分布式梯度裁剪可以减少通信开销，但需要更多的协作和信任。
                在实际应用中，可能需要权衡这些优缺点，选择适合特定场景的技术。
            梯度裁剪是一种用于控制梯度的大小的技术，以防止梯度爆炸问题。其中一种常见的梯度裁剪方法是基于L2范数（Euclidean范数）的裁剪，但还有其他一些方法，包括：
                L2范数裁剪：
                    这是最常见的梯度裁剪方法。它的原理是计算梯度的L2范数，如果L2范数超过了一个预定的阈值（裁剪阈值），就对梯度进行缩放，
                    以确保它不会太大。具体操作是将所有梯度的L2范数限制在一个合理的范围内。
                L1范数裁剪：
                    类似于L2范数裁剪，但是使用的是梯度的L1范数。它将所有梯度的L1范数限制在阈值内。
                均方根（RMS）裁剪：
                    均方根裁剪是一种自适应的梯度裁剪方法，它不需要手动设置裁剪阈值。它计算梯度的均方根，并将其与一个预定的最大均方根值进行比较。
                    如果均方根超过了阈值，就对梯度进行缩放。
                百分位梯度裁剪：
                    这是一种相对较新的方法，它将梯度的百分位数与一个固定的百分位数（通常是99%）进行比较，而不是使用固定的阈值。如果梯度的百分
                    位数超过了固定百分位数的阈值，就进行裁剪。
                阈值裁剪：
                    这是一种最简单的梯度裁剪方法，其中梯度值超过了一个预定的绝对阈值就被截断为阈值。
                每种梯度裁剪方法都有其适用的场景和优劣势。选择哪种方法取决于问题的性质以及模型的需求。通常情况下，L2范数裁剪是最常见的选择，因为它
                相对简单且有效。然而，其他方法也可以在某些情况下提供更好的性能或更精细的控制。在实际应用中，通常需要进行实验和调整，以找到最合适的梯度裁剪方法和阈值。
            '''

    # 积累权重weight_accumulator
    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)                                                                # 求L2范数
        for name, value in local_update.items():
            # 计算L2范数之后还要和DP的阈值
            weight_accumulator[name].add_(value)

    # 更新全局模型
    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants                                              # scale = 10/100
            average_update = scale * sum_update                                                                         # 联邦平均更新
            self.dp_add_noise(average_update)
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)                                                                           # 全局模型+处理过的本地更新
            '''
            中央服务器在联邦学习中使用学习率乘以平均更新来更新全局模型的权重是为了控制全局模型的更新步长。这是为了确保联邦学习中的全局模型不会
            在每轮训练中立即完全采用来自各个参与方的本地更新，从而帮助实现模型的稳定性和收敛性。下面是一些关于为什么使用学习率的重要原因：
                控制更新步长：学习率是一个小的正数，通常远小于1。通过将平均更新与学习率相乘，可以确保每轮训练中的全局模型权重更新相对较小。这
                有助于防止全局模型在每轮训练中发生剧烈波动，从而稳定训练过程。

                平滑收敛：在联邦学习中，不同的参与方可能具有不同的数据分布和数据大小。如果不使用学习率来控制权重更新的幅度，那么来自数据量较
                大参与方的权重更新可能会主导全局模型的变化，而来自数据量较小参与方的权重更新则会被忽略。学习率有助于平衡这些更新，使得模型能
                够更好地收敛到全局最优解。

                避免震荡和发散：使用学习率有助于避免训练过程中的震荡和发散。如果权重更新的步长过大，模型可能会在每轮训练中跳跃到不稳定的状态，
                难以达到收敛。通过缩小权重更新的幅度，可以减少这种风险。

                保护隐私：在联邦学习中，差分隐私等隐私保护技术可能与学习率结合使用，以进一步保护参与方的隐私。学习率的设置可以影响添加到更新
                中的隐私噪音的影响程度。

                综上所述，学习率的使用是为了平衡全局模型的更新幅度，确保稳定的训练过程和合适的收敛行为，以及在某些情况下，保护隐私。不同的学习
                率设置可能适用于不同的联邦学习问题和数据分布，因此通常需要仔细调整学习率来获得最佳性能。
            '''

    # 差分隐私添加噪声，这里需要自己设置高斯噪声的方差
    def dp_add_noise(self, sum_update_tensor: torch.Tensor):                                                            # 添加差分隐私噪声
        if self.params.fl_diff_privacy:
            # 创建一个名为 noised_layer 的新张量，其形状与 sum_update_tensor 相同。这个张量将用于存储添加了噪音的值。
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            # 将 noised_layer 移动到指定的设备，设备由 self.params.device 指定。
            noised_layer = noised_layer.to(self.params.device)
            # 使用正态分布随机生成 noised_layer 中的值，均值为0，标准差为 self.params.fl_dp_noise。这个操作将为 noised_layer 中的每个元素添加随机噪音。
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)



