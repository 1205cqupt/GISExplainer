import matplotlib.pyplot as plt
import torch
import networkx as nx
import itertools
from torch_geometric.utils import k_hop_subgraph
from utils_test import *
import random
import pandas as pd
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

import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch, Dataset, DataLoader
from dig.xgraph.evaluation import XCollector
# 自己写的工具包
from utils_test import *
from itertools import combinations
from numpy.random import default_rng
from Interaction import Interact
from Final_MyFirstTest import InteractShapley
import time
device = torch.device('cpu')
random.seed(1314)

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
    explainer_name = "only_shapley"


    index = -1
    explain_sparsity = config.explain.sparsity
    if explain_sparsity < 0:
        explain_sparsity = None
    node_indices = config.explain.test_node
    # node_indices = [662, 632, 622, 412,
    # .569, 347] # shapes
    # node_indices = [662] # community
    # node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()[-50:] # ba_shapes
    # node_indices=node_indices[:2]
    # node_indices = [662] # community

    x_collector = XCollector()

    start_time = time.time()
    real_fidetiy_and_sparsity = []
    for target_node in node_indices:
        print("explain node {}".format(target_node))
        logits = model(data.x, data.edge_index)
        prediction = logits[target_node].argmax(-1).item()
        explainer = Interact(data, model, target_node=target_node, num_class=num_classes, explainer_name = explainer_name, sparsity= explain_sparsity)
        # result, related_pred = explainer.explainer(data.x, data.edge_index, label=0)
        _, explanation_results, related_preds = explainer(data, target_node=target_node, sparsity= explain_sparsity)
        explanation_results = explanation_results[prediction]
        x_collector.collect_data(explanation_results.colation, related_preds, label=prediction)

        flag = "useful"

        if related_preds[prediction]["origin"] - related_preds[prediction]["maskout"] < 0.2:
            flag = "useless"
            continue

        real_fidetiy_and_sparsity.append(
            (related_preds[prediction]["origin"] - related_preds[prediction]["maskout"], x_collector.sparsity))

        explain_results = {
            'node_idx': target_node,
            'Fidelity_my': related_preds[prediction]["origin"] - related_preds[prediction]["maskout"],
            'Infidelity_my': related_preds[prediction]["origin"] - related_preds[prediction]["masked"],
            'Sparsity': str(x_collector.sparsity),
            'flag': flag,
        }

        save_node_explain_result(config.explain.explain_node_result_save_pth,
                                 file_name='dataset_{}_{}_{}_sparsity{}_results'.format(config.datasets.dataset_name,
                                                                                        config.models.gnn_name,
                                                                                        explainer_name,
                                                                                        explain_sparsity),
                                 explain_result=explain_results)

    end_time = time.time()
    real_fidelity = sum([x[0] for x in real_fidetiy_and_sparsity])/len(real_fidetiy_and_sparsity)
    real_sparsity = sum([x[1] for x in real_fidetiy_and_sparsity])/len(real_fidetiy_and_sparsity)
    print("running time is {}".format(end_time - start_time))
    print("{}模型，{}数据集，{}解释方法".format(config.models.gnn_name, config.datasets.dataset_name, explainer_name))
    print(f'Fidelity: {x_collector.fidelity:.4f}\n',
          f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}\n',
          f'Fidelity_real: {real_fidelity:.4f}\n',
          f'Sparsity_real: {real_sparsity:.4f}')

    # 创建一个数据帧
    results = {
        'Model': config.models.gnn_name,
        'Dataset': config.datasets.dataset_name,
        'Explainer': explainer_name,
        "Fidelity_real": str(real_fidelity),
        "Sparsity_real": str(real_sparsity),
        'Fidelity': str(x_collector.fidelity),
        'Infidelity': str(x_collector.fidelity_inv),
        "set_sparsity": str(explain_sparsity),
        'Sparsity': str(x_collector.sparsity),
        'Running Time': end_time - start_time,
        "test_data": str(node_indices),
    }
    save_all_explain_result(save_pth=config.explain.explain_all_result_save_pth,
                            file_name='dataset_{}_{}_{}_sparsity{}_results'.format(config.datasets.dataset_name,
                                                                        config.models.gnn_name, explainer_name, explain_sparsity),
                            explain_result=results)

    print("Done!")

if __name__ == "__main__":
    import os
    main()