import time

import matplotlib.pyplot as plt
import torch
import networkx as nx
import itertools
from torch_geometric.utils import k_hop_subgraph
from utils_test import *
import random
from tqdm import tqdm
# import os
import torch
import pandas as pd
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
from InteractShapley import InteractShapley
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch, Dataset, DataLoader
from dig.xgraph.evaluation import XCollector
# 自己写的工具包
from utils_test import *
from itertools import combinations
from numpy.random import default_rng
from Interaction import Interact
device = torch.device('cpu')
random.seed(1314)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="config", config_name="test")
def main(config):
    # Set config
    cwd = os.path.dirname(os.path.abspath(__file__))
    config.datasets.dataset_root = os.path.join(cwd, "datasets")
    config.models.gnn_saving_path = os.path.join(cwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")

    config.models.param = config.models.param[config.datasets.dataset_name]
    # config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = config.explainers.explainer_name
    device = torch.device("cpu")
    # 载入数据
    dataset = get_dataset('datasets', config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.y = dataset.data.y.squeeze().long()
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    splitted_dataset = split_dataset(dataset, num_classes)
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)
    data = dataset[0]
    # 载入模型
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            config.datasets.dataset_name,
            f"{config.datasets.dataset_name}_{config.models.gnn_name}_best.pth",
        )
    )["net"]
    model.load_state_dict(state_dict)
    explainer_name = "GNNExplainer"

    ### Display example output
    from dig.xgraph.method import GNNExplainer

    explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
    index = -1
    sparsity = 0.5
    from dig.xgraph.evaluation import XCollector
    x_collector = XCollector(sparsity)
    # 修改的
    node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()[:10]
    nodes_need_explain = torch.tensor(node_indices)
    result_frames = []
    start_time = time.time()

    for i, data in enumerate(dataloader):
        for j, node_idx in enumerate(nodes_need_explain):
            index += 1
            print(f'explain graph {i} node {node_idx}')
            data.to(device)
            if torch.isnan(data.y[0].squeeze()):
                continue
            edge_masks, hard_edge_masks, related_preds = explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx)
            # print(hard_edge_masks.shape)
            # print(related_preds)
            x_collector.collect_data(hard_edge_masks, related_preds, data.y[0].squeeze().long().item())
            explain_results = {
                'node_idx': [node_idx],
                'Fidelity_my': [related_preds[0]["origin"] - related_preds[0]["maskout"]],
                'Fidelity': [str(x_collector.fidelity)],
                'Infidelity': [str(x_collector.fidelity_inv)],
                'Infidelity_my': [related_preds[0]["origin"] - related_preds[0]["masked"]],
                'Sparsity': [str(x_collector.sparsity)],
            }
            df_explain_results = pd.DataFrame(explain_results)
            result_frames.append(df_explain_results)
            print("Done!")
            # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
            # obtain the result: x_processor(data, masks, x_collector)
            if index >= 99:
                break
    end_time = time.time()
    print(config.datasets.dataset_name)
    # 保存数据
    df_combined = pd.concat(result_frames)
    # 文件路径
    folder_name = 'explaine_result'
    os.makedirs(folder_name, exist_ok=True)
    output_file = os.path.join(folder_name, 'dataset_{}_{}_{}_results.csv'.format(config.datasets.dataset_name,
                                                                                  config.models.gnn_name,
                                                                                  explainer_name))
    # 保存数据帧为表格csv，如果已经存在则追加到文件后面
    if not os.path.isfile(output_file):
        df_combined.to_csv(output_file, index=False)
    else:
        with open(output_file, 'a') as file:
            df_combined.to_csv(file, index=False, header=False)
    print("{}模型，{}数据集，{}解释方法".format(config.models.gnn_name, config.datasets.dataset_name, explainer_name))
    print(f'Fidelity: {x_collector.fidelity:.4f}\n',
          f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    folder_name = 'result'
    os.makedirs(folder_name, exist_ok=True)
    output_file = os.path.join(folder_name, 'dataset_{}_{}_{}_results.txt'.format(config.datasets.dataset_name,
                                                                                  config.models.gnn_name,
                                                                                  explainer_name))
    # x_collector.fidelity 的前三位数字
    # 创建一个数据帧
    results = {
        'Model': [config.models.gnn_name],
        'Dataset': [config.datasets.dataset_name],
        'Explainer': [explainer_name],
        'Fidelity': [str(x_collector.fidelity)],
        'Infidelity': [str(x_collector.fidelity_inv)],
        'Sparsity': [str(x_collector.sparsity)],
        'Running Time': [end_time - start_time],
        "test_data": [node_indices]
    }
    df = pd.DataFrame(results)
    # 将数据帧追加到 TXT 文件
    if not os.path.isfile(output_file):
        # 如果文件不存在，创建新文件并保存数据帧
        df.to_csv(output_file, index=False, sep='\t')
    else:
        # 如果文件已存在，将数据帧追加到现有文件
        with open(output_file, 'a') as file:
            df.to_csv(file, index=False, sep='\t', header=False)
    print("数据写入成功。文件路径：", output_file)

if __name__ == "__main__":
    import sys
    main()