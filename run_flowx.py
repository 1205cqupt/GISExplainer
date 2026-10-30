import os
import os.path as osp

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import DataLoader

from dig.xgraph.dataset import SynGraphDataset

from dig.xgraph.models import *

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def split_dataset(dataset):
    indices = []
    num_classes = 4
    train_percent = 0.7
    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset


dataset = SynGraphDataset('./datasets', 'BA_shapes')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.x = dataset.data.x[:, :1]
# dataset.data.y = dataset.data.y[:, 2]
dim_node = dataset.num_node_features

dim_edge = dataset.num_edge_features
# num_targets = dataset.num_classes
num_classes = dataset.num_classes

splitted_dataset = split_dataset(dataset)
splitted_dataset.data.mask = splitted_dataset.data.test_mask
# splitted_dataset.slices['mask'] = splitted_dataset.slices['train_mask']
dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)


model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model.to(device)
check_checkpoints()
ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
state_dict = compatible_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
model.load_state_dict(state_dict)

data = list(dataloader)[0].to(device)
out = model(data.x, data.edge_index)
print(out)


from dig.xgraph.method import GradCAM


explainer = GradCAM(model, explain_graph=False)


# --- Set the Sparsity to 0.5 ---
sparsity = 0.5

from dig.xgraph.evaluation import XCollector

x_collector = XCollector()

### Run explainer on the given model and dataset
index = -1
node_indices = [662,632,622,412,569,347]


for i, data in enumerate(dataloader):
    for j, node_idx in enumerate(node_indices):
        index += 1
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue

        with torch.no_grad():
            walks, masks, related_preds = explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)

            x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())

        # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
        # obtain the result: x_processor(data, masks, x_collector)
        if index >= 99:
            break

    if index >= 20:
        break

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
