"""
默认一件事：那就是输入的batch_size=1
输入：模型和输入样本
功能：更新神经元激活状态图，并能够返回未激活的索引
"""

import torch
import numpy
from collections import defaultdict
import random
from cifar10_resnet import ResNet18
import numpy as np


class NeuronCoverage:
    def __init__(self, model):
        self.model = model

        self.all_layer_name = self._get_all_layer_name()

        self.model_layer_dict = self._init_coverage_table()

    def _init_coverage_table(self):
        model_layer_dict = defaultdict(bool)

        for name, module in self.model.named_modules():
            if name in self.all_layer_name:
                for index in range(module.out_channels):
                    model_layer_dict[(name, index)] = False

        return model_layer_dict

    def _get_all_layer_name(self):
        all_layer_name = []
        # 这里只选择卷积层 因为torch.nn.Linear往往是只有一层 最后接的就直接是分类结果啦 要是选择这个神经元就相当于有目标的对抗攻击
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                all_layer_name.append(name)
        return all_layer_name

    def get_neuron_to_cover(self):
        # 返回一个未覆盖的 layer_name 和 index,如果全覆盖则随机返回
        not_covered = [(layer_name, index) for (layer_name, index), v in self.model_layer_dict.items() if not v]

        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self.model_layer_dict.keys())

        return layer_name, index

    def update_coverage(self, data, threshold=0.75):
        layer_name = self.all_layer_name

        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            scaled = self._scale(output[0])

            for num_neuron in range(scaled.shape[0]):
                if np.mean(scaled[num_neuron, ...]) > threshold \
                        and not self.model_layer_dict[(layer_name[i], num_neuron)]:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = True

    def is_full_coverage(self):
        if False in self.model_layer_dict.values():
            return False
        else:
            return True

    def _scale(self, input, rmax=1, rmin=0):
        # input size (channel,h,w)-->(channel)
        input = input.cpu().detach().numpy()

        input_std = (input - np.min(input)) / (input.max() - input.min())
        input_scaled = input_std * (rmax - rmin) + rmin
        return input_scaled

    def neuron_coverage_rate(self):
        # 统计神经元覆盖率
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        total_neurons = len(self.model_layer_dict)

        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def _get_forward_value(self, data):
        all_out = []

        def forward_hook(module, input, output):
            all_out.append(output)

        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(forward_hook)

        output = self.model(data)

        return all_out


def get_output(model, data, layer_name, index):
    out = []

    def forward_hook(module, input, output):
        out.append(output)

    for name, modules in model.named_modules():
        if name == layer_name:
            modules.register_forward_hook(forward_hook)

    forward_out = model(data)

    # out[0] : (batch_size,channel,h,w)

    index_out = out[0][0, index, ...]

    return index_out


def main():
    model = ResNet18()
    input_ = torch.randn(1,3,32,32)

    aa = NeuronCoverage(model)
    print(aa.all_layer_name)
    # print(aa.model_layer_dict)

    aa.update_coverage(data=input_)

    layer_name,index = aa.get_neuron_to_cover()

    print(layer_name,index)

    input_x = get_output(model,input_,layer_name,index)

    loss = torch.mean(input_x)
    loss.backward()

    print(input_x)


if __name__ == "__main__":
    main()
