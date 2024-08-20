import numpy as np
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from train.train_imp import train_model_and_masker, eval_acc_with_mask, eval_with_mask#, train_epoch, evaluate_network
from nets.gcn_net import BiGCNNet,BiGCNMarker
from data.data import my_load_data # import dataset
import pruning
import copy
import pdb
from utils import EMA
import pickle
import warnings

# Reference: Part of the code is derived from paper:
# Fan, Shaohua, Xiao Wang, Yanhu Mo, Chuan Shi, and Jian Tang. 
# “Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure.” 
# In NeurIPS, 2022. http://arxiv.org/abs/2209.14107.


warnings.filterwarnings("ignore")


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
def train_get_mask(MODEL_NAME, dataset_ori, net_params, things_dict, imp_num, filename, pruned_filename, args):
    t0 = time.time()
    #trainset_ori, valset_ori, testset_ori = dataset_ori.train, dataset_ori.val, dataset_ori.test
    # What is for biased_Test and unbiased_test?
    trainset_ori, valset_ori, biased_testset_ori, unbiased_testset_ori = dataset_ori.id_train, dataset_ori.id_test, dataset_ori.id_test, dataset_ori.ood_test
    print('Experiment on Twitter dataset: ')
    print(len(trainset_ori), len(valset_ori), len(biased_testset_ori), len(unbiased_testset_ori))
    train_loader_ori = DataLoader(trainset_ori, batch_size=net_params["batch_size"], shuffle=True, drop_last=False)
    val_loader_ori = DataLoader(valset_ori, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)
    biased_test_loader_ori = DataLoader(biased_testset_ori, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)
    unbiased_test_loader_ori = DataLoader(unbiased_testset_ori, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # TODO: parameters of the BiGCN is hard coded. 
    model_c = BiGCNNet(768, 64, 64).to(device)
    #masker_c = mask_model(net_params).to(device) 
    masker_c = BiGCNMarker(768, 64, 64).to(device)
    
    model_b = BiGCNNet(768, 64, 64).to(device)
    masker_b = BiGCNMarker(768, 64, 64).to(device)

    train_label = [train.y for train in trainset_ori]

    # EMA yields TypeError: len() of unsized object. 
    sample_loss_ema_c = EMA(torch.LongTensor(train_label), num_classes=10, alpha=0.7)
    sample_loss_ema_b = EMA(torch.LongTensor(train_label), num_classes=10, alpha=0.7) 
    if things_dict is not None:

        trainset_pru, valset_pru, biased_testset_pru, unbiased_testset_pru = things_dict['trainset_pru'], things_dict['valset_pru'], things_dict['biased_testset_pru'], things_dict['unbiased_testset_pru']
        train_loader_pru = DataLoader(trainset_pru, batch_size=net_params["batch_size"], shuffle=True, drop_last=False, collate_fn=dataset_ori.collate)
        val_loader_pru = DataLoader(valset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        biased_test_loader_pru = DataLoader(biased_testset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        unbiased_test_loader_pru = DataLoader(unbiased_testset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)

        rewind_weight_c = things_dict['rewind_weight_c']
        rewind_weight_b = things_dict['rewind_weight_b']
        rewind_weight2 = things_dict['rewind_weight2']
        model_mask_dict_c = things_dict['model_mask_dict_c']
        model_mask_dict_b = things_dict['model_mask_dict_b']
        model_c.load_state_dict(rewind_weight_c)

        model_b.load_state_dict(rewind_weight_b)
        #pruning.pruning_model_by_mask(model_c, model_mask_dict_c)

        #pruning.pruning_model_by_mask(model_b, model_mask_dict_b)

        masker_c.load_state_dict(rewind_weight2)
        
    else:
        trainset_pru = copy.deepcopy(trainset_ori)
        valset_pru = copy.deepcopy(valset_ori)
        biased_testset_pru = copy.deepcopy(biased_testset_ori)
        unbiased_testset_pru = copy.deepcopy(unbiased_testset_ori)

        train_loader_pru = DataLoader(trainset_pru, batch_size=net_params["batch_size"], shuffle=True, drop_last=False,)
        val_loader_pru = DataLoader(valset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)
        biased_test_loader_pru = DataLoader(biased_testset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)
        unbiased_test_loader_pru = DataLoader(unbiased_testset_pru, batch_size=net_params["batch_size"], shuffle=False, drop_last=False)

        rewind_weight_c = copy.deepcopy(model_c.state_dict())

        rewind_weight_b = copy.deepcopy(model_b.state_dict())
        rewind_weight2 = copy.deepcopy(masker_c.state_dict())

    sp_train = pruning.print_pruning_percent(train_loader_ori, train_loader_pru, "train")
    sp_val = pruning.print_pruning_percent(val_loader_ori, val_loader_pru, "val")
    sp_biased_test = pruning.print_pruning_percent(biased_test_loader_ori, biased_test_loader_pru, "biased")
    sp_unbiased_test = pruning.print_pruning_percent(unbiased_test_loader_ori, unbiased_test_loader_pru, "unbiased")

    spa = (sp_train + sp_biased_test + sp_val + sp_unbiased_test) / 4
    spw = pruning.see_zero_rate(model_c)
    optimizer_c = optim.Adam([{'params': model_c.parameters(), 'lr': 0.01}, 
                            {'params': masker_c.parameters(),'lr': 0.01}], weight_decay=0)

    scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, mode='min',factor=0.5, patience=20, verbose=True)               
    optimizer_b = optim.Adam([{'params': model_b.parameters(), 'lr': 0.01}, 
                            {'params': masker_b.parameters(),'lr': 0.01}], weight_decay=0)

    scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='min',factor=0.5, patience=20, verbose=True)    
                                             
    run_time, best_val_acc, best_epoch, update_biased_test_acc, update_unbiased_test_acc  = 0, 0, 0, 0, 0

    best_unbiased_test_acc, best_biased_test_acc = 0, 0
    print("done! cost time:[{:.2f} min]".format((time.time() - t0) / 60))
    best_biased_report, best_unbiased_report = None, None
    model_c_params, model_b_params, masker_c_params = None, None, None
    for epoch in range(args.mask_epochs):
      
        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer, mask_distribution = train_model_and_masker(model_c, masker_c, model_b, masker_b, optimizer_c, optimizer_b, sample_loss_ema_c, sample_loss_ema_b, device, train_loader_pru, epoch, args)
        epoch_val_loss, epoch_val_acc = eval_acc_with_mask(model_c, masker_c, model_b, device, val_loader_pru, epoch, args, val = True)
        _, epoch_biased_test_acc, epoch_biased_eval_report = eval_with_mask(model_c, masker_c, model_b, device, biased_test_loader_pru, epoch, args)     
        _, epoch_unbiased_test_acc, epoch_unbiased_eval_report = eval_with_mask(model_c, masker_c, model_b, device, unbiased_test_loader_pru, epoch, args)     
        scheduler_c.step(epoch_val_loss)

        scheduler_b.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:

            best_val_acc = epoch_val_acc
            update_biased_test_acc = epoch_biased_test_acc
            update_unbiased_test_acc = epoch_unbiased_test_acc
            best_epoch = epoch
            best_masker_state_dict = copy.deepcopy(masker_c.state_dict())
        if epoch_biased_test_acc > best_biased_test_acc:
            best_biased_test_acc = epoch_biased_test_acc
            best_biased_report = epoch_biased_eval_report
        if epoch_unbiased_test_acc > best_unbiased_test_acc:
            best_unbiased_test_acc = epoch_unbiased_test_acc
            best_unbiased_report = epoch_unbiased_eval_report
            model_c_params = copy.deepcopy(model_c.state_dict())
            model_b_params = copy.deepcopy(model_b.state_dict())
            masker_c_params = copy.deepcopy(masker_c.state_dict())

        print('-'*120)
        str1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' + 'Train IMP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Biased Test:[{:.2f}] Update Biased Test:[{:.2f}] Best Biased Test:[{:.2f}]| Unbiased Test:[{:.2f}]  Update Unbiased Test:[{:.2f}] Best Unbiased Test:[{:.2f}]| epoch:[{}] | Time:[{:.2f} min] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]'.format(imp_num,
                        spa * 100,
                        spw * 100,
                        epoch + 1, 
                        args.mask_epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_biased_test_acc * 100, 
                        update_biased_test_acc * 100,
                        best_biased_test_acc * 100,
                        epoch_unbiased_test_acc * 100, 
                        update_unbiased_test_acc * 100,
                        best_unbiased_test_acc * 100,
                        best_epoch,
                        run_time / 60,
                        mask_distribution[0] * 100,
                        mask_distribution[1] * 100,
                        mask_distribution[2] * 100,
                        mask_distribution[3] * 100,
                        mask_distribution[4] * 100) + '\n'
        with open(filename, 'a') as result_file:
            result_file.write(str1)
        result_file.close()
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'Train IMP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Biased Test:[{:.2f}] Update Biased Test:[{:.2f}] Best Biased Test:[{:.2f}]| Unbiased Test:[{:.2f}]  Update Unbiased Test:[{:.2f}] Best Unbiased Test:[{:.2f}] epoch:[{}] | Time:[{:.2f} min] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]'
                .format(imp_num,
                        spa * 100,
                        spw * 100,
                        epoch + 1, 
                        args.mask_epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_biased_test_acc * 100, 
                        update_biased_test_acc * 100,
                        best_biased_test_acc * 100,
                        epoch_unbiased_test_acc * 100, 
                        update_unbiased_test_acc * 100,
                        best_unbiased_test_acc * 100,
                        best_epoch,
                        run_time / 60,
                        mask_distribution[0] * 100,
                        mask_distribution[1] * 100,
                        mask_distribution[2] * 100,
                        mask_distribution[3] * 100,
                        mask_distribution[4] * 100)) 
        print('-'*120)
        
    things_dict = {}
    with open(filename, 'a') as result_file:
        result_file.write(str(epoch_unbiased_test_acc * 100)+'\n')
    result_file.close()

    things_dict['best_biased_report'] = best_biased_report
    things_dict['best_unbiased_report'] = best_unbiased_report
    
    things_dict['model_c_params'] = model_c_params
    things_dict['model_b_params'] = model_b_params
    things_dict['masker_c_params'] = masker_c_params

    print("save data done!")
    t1 = time.time()
    print("INFO: Data Sparsity:[{:.2f}%] time:[{:.2f} min]".format(spa * 100, (t1 - t0)/60))

    return things_dict
    
def main():  
    args = pruning.parser_loader().parse_args()
    pruning.setup_seed(args.seed)
    pruning.print_args(args)  
    
    print(os.getcwd(), args.config)
    with open(os.path.join(os.getcwd(), args.config)) as f:
        config = json.load(f)
    
    DATASET_NAME = 'Twitter'
    MODEL_NAME = 'BiGCN'

    dataset = my_load_data(DATASET_NAME)
    # dataset is input into train_get_mask. dataset has four part: train, val, biased_test, unbiased_test
    # eg. train is a Dataset object, train[i], containing a dgl_graph object and a label. 
    params = config['params']
    params['seed'] = int(args.seed)
    net_params = config['net_params']
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = 768  
    #net_params['in_dim'] = dataset.id_train[0][0].ndata['feat'][0].size(0)
    #net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    net_params['irm_lambda'] = args.irm_lambda
    net_params['in_dim_edge'] = 0
    net_params['n_classes'] = 2

    things_dict = None
    filename = './results/'+args.out_dir +"_"+str(params['seed'])+ '.txt'

    pruned_filename = './pruned_data/'+args.out_dir
    if os.path.exists(filename):
        os.remove(filename)  

    for imp_num in range(1, 2):
    #    args.pa = 0.05
    #    args.pw = 0.2
        things_dict = train_get_mask(MODEL_NAME, dataset, net_params, things_dict, imp_num, filename, pruned_filename, args)
        print('Best biased report: ', things_dict['best_biased_report'])
        print('Best unbiased report: ', things_dict['best_unbiased_report'])  
        torch.save(things_dict['model_c_params'], 'model_c.pth')
        torch.save(things_dict['model_b_params'], 'model_b.pth')
        torch.save(things_dict['masker_c_params'], 'masker_c.pth')
   #     eval_tickets(dataset, net_params, things_dict, imp_num, args)

    
if __name__ == '__main__':
    main()

