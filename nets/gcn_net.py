import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import copy
from mlp_readout_layer import MLPReadout

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# GNN model
'''
Input: data.
    x, edge_index, BU_edge_index, y, root, root_index, seqlen


Input of GraphConv: 
    graphconv(dgl_g, feat, edge_weigh=edge_mask)
'''


class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data, edge_mask=None):
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index, edge_weight=edge_mask)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_mask)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x_mean = scatter_mean(x, data.batch, dim=0)

        return x, x_mean

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data, edge_mask=None):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index, edge_weight=edge_mask)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_mask)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x_mean= scatter_mean(x, data.batch, dim=0)
        return x, x_mean



class BiGCNNet(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BiGCNNet, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)

        self.MLP_layer = th.nn.Linear((out_feats+hid_feats)*4, 2)


    def forward(self, data, edge_mask=None, node_mask=None):
        if node_mask is not None:
            data.x = data.x * node_mask
        _, TD_x = self.TDrumorGCN(data, edge_mask)
        _, BU_x = self.BUrumorGCN(data, edge_mask)
        x = th.cat((BU_x,TD_x), 1)

        #x=self.fc(x)
        #x = F.log_softmax(x, dim=1)
        return x
    
    def loss(self, pred, label):
        criterion = th.nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    

class BiGCNMarker(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BiGCNMarker, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.node_mlp = MLPReadout((hid_feats+out_feats)*2, 1)
        self.edge_mlp = MLPReadout((hid_feats+out_feats)*4, 1)
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, data):
        TD_x, _ = self.TDrumorGCN(data)
        BU_x, _ = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        # x is the node feature set 'h'
        node_score = self.node_score(x)
        edge_score = self.edge_score(x, data.edge_index)
        self.sigmoid = th.nn.Sigmoid()
        
        return edge_score, node_score
    

    def node_score(self, x):
        node_score = self.node_mlp(x)
        node_score = self.sigmoid(node_score)
        
        return node_score

    '''
    Here we have two implementation to filter the edges: 
    (1) Filter on the edge_index, create the BU_edge_index before forward into BiGCN by reversing. 
    --> func edge_score
    (2) Filter on both of the edge_index and BU_edge_index, therefore edge_index & BU_edge_index 
    are not corelated anymore. 
    --> func edge_score_1
    '''
    def edge_score(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        edge_score = th.cat((x[row], x[col]), dim=1)
        edge_score = self.edge_mlp(edge_score)
        edge_score = self.sigmoid(edge_score)
        
        return edge_score
     
    def edge_score_1(self, x, edge_index, BU_edge_index):
        # Return edge_score for both edge_index and BU_edge_index
        row, col = edge_index[0], edge_index[1]
        edge_score = th.cat((x[row], x[col]), dim=1)
        edge_score = self.edge_mlp(edge_score)
        edge_score = self.sigmoid(edge_score)

        BU_row, BU_col = BU_edge_index[0], BU_edge_index[1] 
        BU_edge_score = th.cat((x[BU_row], x[BU_col]), dim=1)
        BU_edge_score = self.edge_mlp(BU_edge_score)
        BU_edge_score = self.sigmoid(BU_edge_score)

        return edge_score, BU_edge_score