import torch
from synthesizers.synthesizer import Synthesizer

# 语义后门暂未实现

class PhysicalSynthesizer(Synthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that represents the physical object in the real scene.
    """
    pattern_tensor = torch.tensor([[1.]])

    # 在图片中加入pattern
    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()                                                                              # 在attack的portion上加入pattern
        batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern
        return

    # 在标签中加入pattern
    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)                                                 # 在attack的portion上加入backdoor_label，这里backdoor_label是8
        return