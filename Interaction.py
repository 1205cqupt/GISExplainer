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
from Final_MyFirstTest import InteractShapley
from top_k_model import top_k
from only_shapley import only_shapley
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch, Dataset, DataLoader
from dig.xgraph.evaluation import XCollector
# 自己写的工具包
from utils_test import *
from itertools import combinations
from numpy.random import default_rng
device = torch.device('cpu')
random.seed(1314)
class Result(Data):
    """继承data类"""
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colation = None
        self.result_edge_index = None

class Interact:
    def __init__(self, data, model, k=2, threshold=0.2, target_node=None,
                 explain_graph = False, num_class = 4, vis = False, save_dir = None,
                 filename = 'example', explainer_name = None , sparsity = None):
        self.data = data
        self.model = model
        self.explainer_name = explainer_name
        self.k = k
        self.threshold = threshold
        self.explain_graph = explain_graph
        self.target_node = target_node
        self.all_edges_shapley_value = []
        self.num_class = num_class
        self.device = torch.device('cpu')
        self.subset, self.sub_edge_index,_,_ = self.get_subgraph(self.data, self.target_node, k = self.k)
        self.sparsity = sparsity
        # self.node_idx_in_subgraph = torch.nonzero(self.subset == self.target_node).squeeze().tolist()

        # saving and visualization
        self.vis = vis
        self.save_dir = save_dir
        self.filename = filename
        self.save = True if self.save_dir is not None else False

        self.calculate_all_edges_shapley = None
    def get_subgraph(self,data,target_node,k):
        subset, edge_index, _, _ = k_hop_subgraph(target_node, k, data.edge_index, relabel_nodes=True)
        return subset, edge_index, _, _

    # def GnnNetsGC2valueFunc(self, gnnNets, target_class):
    #     def value_func(batch):
    #         with torch.no_grad():
    #             logits = gnnNets(data=batch)
    #             probs = F.softmax(logits, dim=-1)
    #             score = probs[:, target_class]
    #         return score
    #
    #     return value_func

    def GnnNetsGC2valueFunc(self, gnnNets_NC, node_idx, target_class):
        def value_func(data):
            with torch.no_grad():
                logits = gnnNets_NC(data=data)
                probs = F.softmax(logits, dim=-1)
                # select the corresponding node prob through the node idx on all the sampling graphs
                batch_size = data.batch.max() + 1
                probs = probs.reshape(batch_size, -1, probs.shape[-1])
                score = probs[:, node_idx, target_class]
                return score

        return value_func

    def gnn_score(self, coalition: list, data: Data, value_func: str, subgraph) -> torch.Tensor:
        """ the value of subgraph with selected nodes """
        num_nodes = data.num_nodes
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
        mask[coalition] = 1.0
        ret_x = mask.view(-1, 1)
        mask_data = Data(x=ret_x, edge_index=subgraph.edge_index)
        mask_data = Batch.from_data_list([mask_data])
        score = value_func(mask_data)
        # get the score of predicted class for graph or specific node idx
        return score.item()

    def sparsity_function(self, result , data, subgraph_building_method='zero_filling'):
        if subgraph_building_method == 'zero_filling':
            return 1.0 - result.edge_index.shape[1] / data.edge_index.shape[1]

    @torch.no_grad()
    # 重写
    def explainer(self, x, edge_index, label, sparsity = None):
        with torch.no_grad():
            probs = self.model(x, edge_index).squeeze().softmax(dim=1)
            prediction = probs[self.target_node].argmax(-1).item() # ground truth

        subgraph_x = torch.tensor([i for i in range(len(self.subset))])
        subgraph = Data(x=subgraph_x, edge_index=self.sub_edge_index)
        if self.explainer_name == 'only_shapley':
            method = only_shapley(self.data, self.model, target_node=self.target_node,
                                    label = label, subset = self.subset, sub_edge_index = self.sub_edge_index, sparsity = self.sparsity)
        elif self.explainer_name == 'top_k':
            method = top_k(self.data, self.model, target_node=self.target_node,
                           label=label, subset=self.subset, sub_edge_index=self.sub_edge_index ,sparsity= self.sparsity)
        elif self.explainer_name == "InteractShapley":
            method = InteractShapley(self.data,self.model,target_node = self.target_node,
                                     label = label, subset = self.subset, sub_edge_index= self.sub_edge_index , sparsity= self.sparsity)

        results = method.get_explain_subgraph(self.subset, self.sub_edge_index) # 得到解释子图的边

        # 控制稀疏度
        if sparsity is not None:
            split_point = int(self.sub_edge_index.shape[1] * (1 - sparsity)) + 1
            results = results[:split_point]

        subgraph_nodes = list(set([node for edge in results for node in edge]))
        subgraph_edge_index = torch.tensor(results, dtype=torch.long).t().contiguous()
        x = torch.tensor(subgraph_nodes, dtype=torch.float).view(-1, 1)
        # 创建Data对象
        result_data = Data(x= x, edge_index=subgraph_edge_index)
        result = Result(x=x, edge_index=subgraph_edge_index)
        result.colation = subgraph_nodes
        result.result_edge_index = results

        # 重要结构
        node_idx = torch.nonzero(self.subset == self.target_node).squeeze().tolist()
        value_func = self.GnnNetsGC2valueFunc(gnnNets_NC = self.model, node_idx = node_idx, target_class=label)

        masked_node_list = [node for node in range(subgraph.x.shape[0])
                          if node in subgraph_nodes]
        # 移除重要结构
        maskout_node_list = [node for node in range(subgraph.x.shape[0]) if node not in subgraph_nodes]
        if not self.explain_graph:
            maskout_node_list += [node_idx]

        masked_score = self.gnn_score(masked_node_list,
                                      subgraph,
                                      value_func=value_func,
                                      subgraph=result_data)

        # 重新计算删除重要结构后的子图
        subgraph_edges = subgraph.edge_index.t().tolist()
        result_edges = result.edge_index.t().tolist()
        subgrpah_edges_set = set([tuple(edge) for edge in subgraph_edges])
        result_edges_set = set([tuple(edge) for edge in result_edges])
        maskout_edges = subgrpah_edges_set - result_edges_set
        complement_edges = list(maskout_edges)
        maskout_edge_index = torch.tensor(complement_edges, dtype=torch.long).t().contiguous()
        subgraph_nodes = list(set([node for edge in complement_edges for node in edge]))
        x = torch.tensor(subgraph_nodes, dtype=torch.float).view(-1, 1)
        result_maskout = Data(x=x, edge_index=maskout_edge_index)

        maskout_score = self.gnn_score(maskout_node_list,
                                  subgraph,
                                  value_func=value_func,
                                  subgraph=result_maskout)

        sparsity_score = self.sparsity_function(result, subgraph)

        related_pred = {'masked': masked_score,
                        'maskout': maskout_score,
                        'origin': probs[self.target_node, label].item(),
                        'sparsity': sparsity_score}

        return result, related_pred


    def __call__(self, data, target_node, max_node = 5, sparsity = None):
        labels = tuple(label for label in range(self.num_class))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        related_preds = []
        explanation_results = []

        # saved_results = None
        # if self.save:
        #     if os.path.isfile(os.path.join(self.save_dir, f"{self.filename}.pt")):
        #         saved_results = torch.load(os.path.join(self.save_dir, f"{self.filename}.pt"))

        for label_idx, label in enumerate(ex_labels):
            results, related_pred = self.explainer(data.x, data.edge_index,
                                                 label = label, sparsity = sparsity)
            related_preds.append(related_pred)
            explanation_results.append(results)

        # if self.save:
        #     torch.save(explanation_results,
        #                os.path.join(self.save_dir, f"{self.filename}.pt"))
        return None, explanation_results, related_preds




if __name__ == "__main__":
    dataset = get_dataset('datasets', 'ba_community')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    splitted_dataset = split_dataset(dataset, num_classes)
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)
    data = dataset[0]

    # 导入模型 gin
    # model = GIN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    # model.to(device='cpu')
    # check_checkpoints()
    # # 注意，这里后面进行对比实验的时候需要扩充,接入hyper库进行管理
    # ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GIN_2l', '0', 'GIN_2l_best.ckpt')
    from gnnNets import GCNNet
    # 导入模型 gcn
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device='cpu')
    check_checkpoints()
    # 注意，这里后面进行对比实验的时候需要扩充,接入hyper库进行管理
    ckpt_path = osp.join('checkpoints', 'ba_community',  'ba_community_gcn_best.pth')
    state_dict = torch.load(ckpt_path)['net']

    # state_dict = compatible_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

    model.load_state_dict(state_dict)
    index = -1
    node_indices = [662,632,622,412,569,347,455]
    # node_indices = [666]
    x_collector = XCollector()
    for target_node in node_indices:
        print("explain node {}".format(target_node))
        logits = model(data.x, data.edge_index)
        prediction = logits[target_node].argmax(-1).item()
        explainer = Interact(data, model, target_node=target_node, num_class=4 , explainer_name="InteractShapley")
        # result, related_pred = explainer.explainer(data.x, data.edge_index, label=0)
        _, explanation_results, related_preds = explainer(data, target_node=target_node)

        explanation_results = explanation_results[prediction]

        x_collector.collect_data(explanation_results.colation, related_preds, label=prediction)
        print("Done!")

    print(f'Fidelity: {x_collector.fidelity:.4f}\n',
          f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')