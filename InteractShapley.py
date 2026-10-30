import matplotlib.pyplot as plt
import torch
import networkx as nx
import itertools
from torch_geometric.utils import k_hop_subgraph
from utils_test import *
import random
random.seed(1314)
from tqdm import tqdm
# import os
import torch
import concurrent.futures
from itertools import combinations
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from gstarx import GStarX
from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils
import torch_geometric.utils as utils
from torch_geometric.data import  DataLoader
import math
import os
import os.path as osp
from dig.xgraph.models import *
import InteractShapley as InteractShapley
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
# 自己写的工具包
from utils_test import *
device = torch.device('cpu')
from itertools import combinations
from numpy.random import default_rng

random.seed(1314)
class InteractShapley:
    def __init__(self, data, model, k=2, threshold=0.2, target_node=None, label = None, subset = None, sub_edge_index = None):
        self.data = data
        self.model = model
        self.model.eval()
        self.k = k
        self.threshold = threshold
        self.target_node = target_node
        self.all_edges_shapley_value = []

        self.label = label
        self.subset = subset
        self.sub_edge_index = sub_edge_index

    def edge_shapley_value(self, src, dst, need_remove = None, alliance_edges = None, num_samples=100):
        # 获取k跳子图及节点和边
        subset, edge_index, _, _ = k_hop_subgraph(self.target_node, self.k, self.data.edge_index, relabel_nodes=True)
        # 保存边的夏普利值
        shapley_value = 0.0

        if not need_remove and not alliance_edges:
            remaining_edge_index = edge_index[:, (edge_index[0] != src) | (edge_index[1] != dst)]
        elif need_remove and not alliance_edges:
            remaining_edge_index = edge_index[:, (edge_index[0] != need_remove[0]) | (edge_index[1] != need_remove[1])]
        elif alliance_edges:
            remaining_edge_index = edge_index
            for edge in alliance_edges:
                remaining_edge_index = remaining_edge_index[:,
                                       (remaining_edge_index[0] != edge[0]) | (remaining_edge_index[1] != edge[1])]
            # 把原来的边再加进去
            remaining_edge_index = torch.cat([remaining_edge_index, torch.tensor([[src], [dst]])], dim=1)

        target_node_index = get_target_node_index(self.target_node, subset)

        edge_nums = edge_index.shape[1]
        node_nums = subset.shape[0]
        # 创建一个节点数相关的0-node_nums的图的节点特征
        subgraph_x = torch.ones((node_nums, 1))
        # 计算完整子图的目标值
        with torch.no_grad():
            for _ in range(num_samples):
                # 随机选择边的子集
                subset_size = random.randint(1, remaining_edge_index.shape[1])
                subset_indices = random.sample(range(remaining_edge_index.shape[1]), subset_size)
                # 再把原来的边加进去
                subset_edge_index = remaining_edge_index[:, subset_indices]
                subset_edge_index_cat_org = torch.cat([subset_edge_index, torch.tensor([[src], [dst]])], dim=1)

                target_logits = self.model(subgraph_x, subset_edge_index_cat_org)
                probs = torch.nn.functional.softmax(target_logits, dim=1)
                label = self.label
                target_value = probs[target_node_index][label]

                subset_logits = self.model(subgraph_x, subset_edge_index)
                sub_probs = torch.nn.functional.softmax(subset_logits, dim=1)
                subset_value = sub_probs[target_node_index][label]

                shapley_value += (target_value - subset_value) / (math.factorial(edge_nums - subset_size - 1) * math.factorial(subset_size) / math.factorial(edge_nums))
        # 计算平均值
        shapley_value = shapley_value / num_samples
        shapley_value = shapley_value.item()

        return shapley_value

    def calculate_all_edges_shapley(self):
        # 计算总边数
        subset, edge_index, = self.subset, self.sub_edge_index
        total_edges = edge_index.shape[1]
        # 初始化进度条
        with tqdm(total=total_edges) as pbar:
            # 遍历图中的每一条边
            for i in range(total_edges):
                src = edge_index[0, i]
                dst = edge_index[1, i]
                # 对这条边计算夏普利值
                score = self.edge_shapley_value(src=src, dst=dst)
                self.all_edges_shapley_value.append(((src.item(), dst.item()), score))
                # 更新进度条
                pbar.update(1)
        return self.all_edges_shapley_value

    def alliance_shapley_value(self, alliance_edges, num_samples = 20):
        # 计算联盟的夏普利值
        # 获取k跳子图及节点和边
        subset, edge_index, = self.subset, self.sub_edge_index
        subgraph_node_features = self.data.x[subset]
        target_node_index = torch.nonzero(subset == self.target_node).squeeze().tolist()

        # 保存联盟的夏普利值
        shapley_value = 0.0

        # 移除联盟的边后的边集
        remaining_edge_index = edge_index

        for edge in alliance_edges:
            src, dst = edge
            remaining_edge_index = remaining_edge_index[:,
                                   (remaining_edge_index[0] != src) | (remaining_edge_index[1] != dst)]
        # 计算完整子图的目标值
        with torch.no_grad():

            # TMC-Shapley 采样 快了五~六倍
            for _ in range(num_samples):
                # 随机选择边的子集

                subset_size = random.randint(2, max(1, remaining_edge_index.shape[1]))
                if remaining_edge_index.shape[1] < subset_size:
                    subset_indices = random.sample(range(remaining_edge_index.shape[1]), remaining_edge_index.shape[1])
                else:
                    subset_indices = random.sample(range(remaining_edge_index.shape[1]), subset_size)
                subset_edge_index = remaining_edge_index[:, subset_indices]
                # 再把合成的加进去
                subset_edge_index_cat = torch.cat([subset_edge_index, torch.tensor([[src], [dst]])], dim=1)
                target_logits = self.model(subgraph_node_features, subset_edge_index_cat)
                probs = torch.nn.functional.softmax(target_logits, dim=1)
                label = self.label
                target_value = probs[target_node_index][label]

                subset_logits = self.model(subgraph_node_features, subset_edge_index)
                sub_probs = torch.nn.functional.softmax(subset_logits, dim=1)
                subset_value = sub_probs[target_node_index][label]

                shapley_value += (target_value - subset_value) / subset_size

            # 计算平均值
        shapley_value /= num_samples
        return shapley_value

    def shapley_interactions_monte_carlo(self, alliance_edges, k=2, monte_carlo_samples = 20):
        BMAX = []
        BMIN = []

        for _ in range(monte_carlo_samples):  # 这里采样代表着这个集合内所有的组合情况
            random_subset = random.sample(alliance_edges, random.randint(2, len(alliance_edges)))
            if len(random_subset) == 2:
                shapley_value =self.shapley_interactions_two_elem(random_subset)
            else:
                shapley_value = self.alliance_shapley_value(random_subset)

            if shapley_value > 0:
                BMAX.append(shapley_value)
            elif shapley_value < 0:
                BMIN.append(shapley_value)

        interaction = sum(BMAX) - sum(BMIN) # T([A]) = Bmax([A]) - Bmin([A])
        return interaction

    def shapley_interactions_two_elem(self, allinance_edges):
        marginal_contribution = 0
        edge1, edge2 = allinance_edges
        marginal_contribution += self.edge_shapley_value(edge1[0], edge1[1], need_remove=edge2)
        marginal_contribution += self.edge_shapley_value(edge2[0], edge2[1], need_remove=edge1)

        return self.alliance_shapley_value(allinance_edges) - marginal_contribution

    def shapley_interactions(self, allinance_edges, k=2, monte_carlo_samples=50):
        # 使用蒙特卡洛方法估计 Shapley 值
        if len(allinance_edges) > 2:
            return self.shapley_interactions_monte_carlo(allinance_edges, k, monte_carlo_samples)
        else:
            return self.shapley_interactions_two_elem(allinance_edges)

    def is_alliance_connected(self, alliance_edges):
        alliance_graph = nx.Graph(alliance_edges)
        return nx.is_connected(alliance_graph)


    def get_explain_subgraph(self, subset, edge_index, max_node = 5):
        alliance_edges = []
        max_shapley_value = -1000000
        if not alliance_edges:
            candidate_edges = find_edges_starting_from_target(edge_index, self.target_node, subset)
            for edge in candidate_edges:
                if type(edge[0]) is int:
                    edge_key = (torch.tensor(edge[0]), torch.tensor(edge[1]))
                else:
                    edge_key = (edge[0], edge[1])
                for saved_edge, shapley_value in self.all_edges_shapley_value:
                    if saved_edge == edge_key:
                        shapley_value = shapley_value
                        if shapley_value > max_shapley_value:
                            max_shapley_value = shapley_value
                            alliance_edges = [saved_edge]
                            used_edges = [saved_edge]
        max_shapley_adjacent_value = -1000000
        while True:
            found_new_edge = False

            max_shapley_adjacent_edges = []
            # temp_adjacent_edges = alliance_edges.copy()
            # for edge in alliance_edges:
            edge = alliance_edges[-1]
            target_node_compute = edge[1]
            candidate_edges = find_adjacent_edges(edge_index, target_node_compute)
            for edge_1 in candidate_edges:
                if edge_1 in used_edges:
                    continue
                temp_alliance_edges = alliance_edges.copy()
                temp_alliance_edges.append(edge_1)
                interaction = self.shapley_interactions(temp_alliance_edges)
                if interaction > max_shapley_adjacent_value:
                    max_shapley_adjacent_value = interaction
                    max_shapley_adjacent_edges = temp_alliance_edges

                    found_new_edge = True

            if found_new_edge:
                alliance_edges = max_shapley_adjacent_edges
                used_edges.extend(max_shapley_adjacent_edges)

            else:
                break
            nodes = set()
            for edge in alliance_edges:
                nodes.add(edge[0])
                nodes.add(edge[1])
            num_nodes = len(nodes)

            if num_nodes == max_node:
                break
        return alliance_edges

if __name__ == "__main__":
    import time
    dataset = get_dataset('datasets', 'ba_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    splitted_dataset = split_dataset(dataset, num_classes)
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)
    print(dataset.data.x.shape)
    print(dataset.data.edge_index.shape)

    # 导入模型
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device='cpu')
    check_checkpoints()
    # 注意，这里后面进行对比实验的时候需要扩充,接入hyper库进行管理
    ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
    state_dict = compatible_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

    model.load_state_dict(state_dict)

    time_start = time.time()
    metrics = {}
    # test_indice = torch.where(splitted_dataset.data.test_mask == True)[0][-2:-1]  # 测试集的节点
    # test_indice = [10,15,67,247,105,199,0,83,286,347,412,569,669]
    # test_indice = [314,317,623]
    # test_indice = [662,632,622,412,569,347]
    test_indice = [662]
    for i, data in enumerate(dataloader):

        data.to(device)
        for target_node in test_indice:
            target_node = torch.tensor([target_node]).to(device)

            logtis = model(data.x, data.edge_index)
            probs = torch.nn.functional.softmax(logtis, dim=1)
            label = probs[target_node][0].argmax().item()

            # 获取目标节点的k跳子图 这个可以不用写了
            subset, edge_index, _, _ = k_hop_subgraph(
                target_node, 2, data.edge_index, relabel_nodes=True)

            target_node_index = torch.nonzero(subset == target_node).squeeze().tolist()
            subgraph_node_features = data.x[subset]
            subgraph_data = Data(x=data.x[subset], edge_index=edge_index)
            print(subgraph_data)
            print(target_node)

            # 创建ShapleyValueCalculator类的实例
            test_interact_shapley = InteractShapley(subgraph_data, model, k=2, threshold=0.2, target_node=target_node, label=label, subset=subset, sub_edge_index=edge_index)

            # 将数据传递给ShapleyValueCalculator类
            test_interact_shapley.data = data
            test_interact_shapley.model = model
            test_interact_shapley.target_node = target_node
            test_interact_shapley.calculate_all_edges_shapley()
            test_interact_shapley.get_explain_subgraph(subset, edge_index, max_node=5)


