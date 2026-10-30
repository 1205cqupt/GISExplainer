import os
import os.path as osp

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import DataLoader

from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.dataset import BA_LRP

from dig.xgraph.models import *

from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils
import torch_geometric.utils as utils
from torch_geometric.data import  DataLoader
import os
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
import time
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

    from dig.xgraph.method import SubgraphX

    explainer = SubgraphX(model, num_classes=num_classes, device=device,
                          explain_graph=False, reward_method='nc_mc_l_shapley')

    # --- Create data collector and explanation processor ---
    from dig.xgraph.evaluation import XCollector

    x_collector = XCollector()
    index = -1
    from dig.xgraph.method.subgraphx import find_closest_node_result
    max_nodes = 5

    # --- Create data collector and explanation processor ---
    from dig.xgraph.evaluation import XCollector

    x_collector = XCollector()

    index = -1
    # node_indices = torch.where(datasets[0].test_mask * datasets[0].y != 0)[0].tolist()
    node_indices = [662,632,622,412,569,347]
    # node_indices = [455]
    from dig.xgraph.method.subgraphx import PlotUtils
    from dig.xgraph.method.subgraphx import find_closest_node_result


    # plotutils = PlotUtils(dataset_name='ba_shapes', is_show=True)
    # explainer.visualization(explanation_results,
    #                         max_nodes=max_nodes,
    #                         plot_utils=plotutils,
    #                         y=data.y)

    node_indices = [648]
    # node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()[-20:]
    # node_indices = [662, 632, 622, 412, 569, 347]
    # node_indices = node_indices[:1]

    #
    # print(node_indices)
    # start_time = time.time()
    # for node_idx in node_indices:
    #     index += 1
    #     print(f'explain graph node {node_idx}')
    #     data.to(device)
    #
    #     if torch.isnan(data.y[0].squeeze()):
    #         continue
    #
    #     logits = model(data.x, data.edge_index)
    #     prediction = logits[node_idx].argmax(-1).item()
    #
    #     _, explanation_results, related_preds =   explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=max_nodes)
    #
    #     explanation_results = explanation_results[prediction]
    #     explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
    #     result = find_closest_node_result(explanation_results, max_nodes=max_nodes)
    #     # plotutils = PlotUtils(dataset_name='ba_shapes', is_show=True)
    #     # explainer.visualization(explanation_results,
    #     #                         max_nodes=max_nodes,
    #     #                         plot_utils=plotutils,
    #     #                         y=data.y)
    #
    #     x_collector.collect_data(result.coalition, related_preds, label=prediction)
    #     if index >= 20:
    #         break
    # explainer_name = "SubgraphX"
    # end_time = time.time()
    #
    # print("{}模型，{}数据集，{}解释方法".format(config.models.gnn_name, config.datasets.dataset_name, explainer_name))
    # print("running time is {}".format(end_time - start_time))
    # print(f'Fidelity: {x_collector.fidelity:.4f}\n',
    #       f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
    #       f'Sparsity: {x_collector.sparsity:.4f}')


    # 测试内容
    target_node = 648
    subset, sub_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(target_node, 2, data.edge_index, relabel_nodes=True)
    target_node_idx = torch.where(subset == target_node)[0].item()
    print(subset)
    print(sub_edge_index)
    coalitions = list(set(sub_edge_index[0].tolist() + sub_edge_index[1].tolist()))[:-4]
    print("coalitions", coalitions)
    sub_edge_index = torch.tensor([(n_frm, n_to) for (n_frm, n_to) in sub_edge_index.T
                                   if n_frm.item() in coalitions and n_to.item() in coalitions], dtype=torch.long).T
    print(sub_edge_index)
    num_nodes = len(subset)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
    mask[coalitions] = 1.0
    ret_x = mask.view(-1, 1)

    subdata = Data(x = ret_x, edge_index=sub_edge_index, y=data.y[subset])

    with torch.no_grad():
        logits = model(subdata.x, subdata.edge_index)
        prediction1 = F.softmax(logits, dim=1)[target_node_idx]
        print(prediction1)


    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        prediction = F.softmax(logits, dim=1)[target_node]
        print(prediction)


if __name__ == "__main__":
    import sys
    main()