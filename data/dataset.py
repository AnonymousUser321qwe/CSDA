import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import itertools
import torch
import torch.utils.data
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from torch_geometric.data import Data, DataLoader
import random

# TODO: Convert the bigcn_small_dgl to __getitem__ based. (Do NOT use _prepare() method.)

class BiGCNDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_ids, droprate=0.2):
        self.data_path = data_path
        self.data_ids = data_ids
        self.tddroprate = droprate
        self.budroprate = droprate
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, index):
        
        id =self.data_ids[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        #print(type(data['seqlen']), data['seqlen'])
        #seqlen = torch.LongTensor([(data['seqlen'])]).squeeze()
        seqlen = torch.LongTensor(np.array(data['seqlen']))
        #x_features=torch.tensor([item.detach().numpy() for item in list(data['x'])],dtype=torch.float32)
        #x_features = torch.stack([torch.from_numpy(item.detach().cpu().numpy()) for item in list(data['x'])]).type(torch.float32)
        x_features_numpy = np.array([item.detach().numpy() for item in list(data['x'])])
        x_features = torch.tensor(x_features_numpy, dtype=torch.float32)  #  <---Copilot's efficient suggestion

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.from_numpy(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]), seqlen=seqlen)


class bigcn_disc_dataset1(torch.utils.data.Dataset):
    # This version is for splited training/valid/test dataset. 
    def __init__(self, DATASET_NAME ,data_dir, id_train, id_valid, id_test, ood_train, ood_valid, ood_test):
        self.id_train, self.id_valid, self.id_test = BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/'+DATASET_NAME+'graph/'), data_ids=id_train), \
                                                     BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/'+DATASET_NAME+'graph/'), data_ids=id_valid), \
                                                     BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/'+DATASET_NAME+'graph/'), data_ids=id_test)

        self.ood_train, self.ood_valid, self.ood_test = BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/'+ DATASET_NAME + 'graph/'), data_ids=ood_train), \
                                                        BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/'+ DATASET_NAME + 'graph/'), data_ids=ood_valid), \
                                                        BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/'+ DATASET_NAME + 'graph/'), data_ids=ood_test)



class bigcn_disc_dataset(torch.utils.data.Dataset):
    # Large dataset, composed of several small datasets. 
    def __init__(self, data_name, data_dir, in_ids, ood_ids):
        self.id_train_ids, self.id_test_ids = self.split_train_test(in_ids)
        self.ood_test_ids, self.ood_train_ids = self.split_train_test(ood_ids)

        if data_name == 'Twitter':
            self.id_train = BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/Twittergraph/'), data_ids=self.id_train_ids)
            self.id_test = BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/Twittergraph/'), data_ids=self.id_test_ids)
            self.ood_train = BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/Twittergraph/'), data_ids=self.ood_train_ids) 
            self.ood_test = BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/Twittergraph/'), data_ids=self.ood_test_ids)
        if data_name == 'Weibo':
            self.id_train = BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/Weibograph/'), data_ids=self.id_train_ids)
            self.id_test = BiGCNDataset(data_path=os.path.join(data_dir, 'in-domain/Weibograph/'), data_ids=self.id_test_ids)
            self.ood_train = BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/Weibograph/'), data_ids=self.ood_train_ids) 
            self.ood_test = BiGCNDataset(data_path=os.path.join(data_dir, 'out-of-domain/Weibograph/'), data_ids=self.ood_test_ids)
        
    def split_train_test(self, data_ids, test_ratio=0.2, contain_valid=False):
        length = len(data_ids)
        train_split, test_split = data_ids[ :int((1-test_ratio)*length)], data_ids[int((1-test_ratio)*length):]
        return train_split, test_split
    '''
    # Form a mini batch from a given list of samples = [(graph, label)]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels))
        
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            #graphs[idx].edata['feat'] = graph.edata['feat'].float()
        #batched_graph = dgl.batch(graphs)
        
        return graphs #, labels
    '''
 

if __name__ == '__main__':
    data_dir = '../data/'
    print(os.getcwd())
    weibo_id_train, weibo_id_valid, weibo_id_test = np.load(os.path.join(data_dir, 'weibo_id_712/weibo_train_fold0.npy')), \
                                                    np.load(os.path.join(data_dir, 'weibo_id_712/weibo_valid_fold0.npy')), \
                                                    np.load(os.path.join(data_dir, 'weibo_id_712/weibo_test_fold0.npy'))
    weibo_ood_train, weibo_ood_valid, weibo_ood_test = np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_train_fold0.npy')), \
                                                        np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_valid_fold0.npy')), \
                                                        np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_test_fold0.npy'))
    
    whole_dgl_dataset = bigcn_disc_dataset1('Weibo', data_dir, weibo_id_train, weibo_id_valid, weibo_id_test, \
                                        weibo_ood_train, weibo_ood_valid, weibo_ood_test)

    id_train, id_valid, id_test, ood_test = whole_dgl_dataset.id_train, whole_dgl_dataset.id_valid, \
                                            whole_dgl_dataset.id_test, whole_dgl_dataset.ood_test


    for train in tqdm(id_train):
        y = train.y