import matplotlib.pyplot as plt
import torch
import networkx as nx
import itertools
from torch_geometric.utils import k_hop_subgraph
from utils_test import *
import random
import math
import numpy as np
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
class top_k:
    def __init__(self, data, model, k=2, threshold=0.2, target_node=None, label = None, subset = None, sub_edge_index = None, sparsity = None):
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

        self.sparsity = sparsity
        self.calculate_all_edges_shapley()

    def edge_shapley_value(self, src, dst, need_remove = None, alliance_edges = None, num_samples=20):
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

                shapley_value += (target_value - subset_value) * (math.factorial(edge_nums - subset_size - 1) * math.factorial(subset_size) / math.factorial(edge_nums))
        # 计算平均值
        shapley_value = shapley_value/num_samples*100
        shapley_value = shapley_value.item()

        return shapley_value

    def calculate_all_edges_shapley(self):
        # 用不着了，可以删了
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

    def get_explain_subgraph(self, subset, edge_index):
        # 获取所有边的夏普利值
        all_shpley_value = self.all_edges_shapley_value
        all_shpley_value = sorted(all_shpley_value, key=lambda x: x[1], reverse=True)
        # 选择夏普利值最大的前20%的边
        top_k_edges = all_shpley_value[:int(1 - len(all_shpley_value) * self.sparsity)]
        alliance = []
        for edge in top_k_edges:
            alliance.append(edge[0])
        return alliance

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
            test_interact_shapley = top_k(subgraph_data, model, k=2, threshold=0.2, target_node=target_node, label=label, subset=subset, sub_edge_index=edge_index)

            # 将数据传递给ShapleyValueCalculator类
            test_interact_shapley.data = data
            test_interact_shapley.model = model
            test_interact_shapley.target_node = target_node
            test_interact_shapley.label = label
            test_interact_shapley.subset = subset

            # 计算所有边的夏普利值
            all_edges_shapley = test_interact_shapley.calculate_all_edges_shapley()
            # 获取前20%的边
            top_k_edges = test_interact_shapley.get_explain_subgraph(subset=subset, edge_index=edge_index)
            print(top_k_edges)