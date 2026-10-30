import torch
subset = [11,33,4,555]
node_nums = len(subset)
node_nums =
subgraph_x = torch.tensor(1).view(node_nums, 1)
subgraph_edge_index = torch.tensor([[0, 1, 1, 2],
                                    [1, 0, 2, 1]], dtype=torch.long)
print(subgraph_x)