""" File to load dataset based on user control from main file
"""
# from data.superpixels import SuperPixDataset, SuperPixDatasetDGL
from data.dataset import *

def my_load_data(DATASET_NAME, data_dir='../data/'):
    if DATASET_NAME == 'Twitter':
        twitter_in_ids = np.load('../data/in-domain/twi_id_ids.npy')
        twitter_out_ids = np.load('../data/out-of-domain/twi_ood_ids.npy')
        whole_dgl_dataset = bigcn_disc_dataset(DATASET_NAME, data_dir=data_dir, in_ids=twitter_in_ids, ood_ids=twitter_out_ids)
        return whole_dgl_dataset
    
    if DATASET_NAME == 'Weibo':
        weibo_in_ids = np.load('../data/in-domain/weibo_id_ids.npy')
        weibo_out_ids = np.load('../data/out-of-domain/weibo_ood_ids.npy')
        whole_dgl_dataset = bigcn_disc_dataset(DATASET_NAME, data_dir=data_dir, in_ids=weibo_in_ids, ood_ids=weibo_out_ids)
        return whole_dgl_dataset

    print('Dataset not found')
    return None
    


def load_712_data(DATASET_NAME, data_dir='../data/'):
    print('Load712 data from: ', DATASET_NAME)
    if DATASET_NAME == 'Twitter':
        # prev version: Get in- out-of- distribution data. Split to both of them into 8:2
        # load the local 712 ids, load the dataset accoding to the ids. 
        # TODO: MAYBE NOT HARDCODING?
        twitter_id_train, twitter_id_valid, twitter_id_test = np.load(os.path.join(data_dir, 'twitter_id_712/twitter_train_fold0.npy')), \
                                                              np.load(os.path.join(data_dir, 'twitter_id_712/twitter_valid_fold0.npy')), \
                                                              np.load(os.path.join(data_dir, 'twitter_id_712/twitter_test_fold0.npy'))
        twitter_ood_train, twitter_ood_valid, twitter_ood_test = np.load(os.path.join(data_dir, 'twitter_ood_712/twitter_train_fold0.npy')), \
                                                                 np.load(os.path.join(data_dir, 'twitter_ood_712/twitter_valid_fold0.npy')), \
                                                                 np.load(os.path.join(data_dir, 'twitter_ood_712/twitter_test_fold0.npy'))
        whole_dgl_dataset = bigcn_disc_dataset1(DATASET_NAME, data_dir, twitter_id_train, twitter_id_valid, twitter_id_test, \
                                                twitter_ood_train, twitter_ood_valid, twitter_ood_test) 
        return whole_dgl_dataset

    if DATASET_NAME == 'Weibo':
        weibo_id_train, weibo_id_valid, weibo_id_test = np.load(os.path.join(data_dir, 'weibo_id_712/weibo_train_fold0.npy')), \
                                                        np.load(os.path.join(data_dir, 'weibo_id_712/weibo_valid_fold0.npy')), \
                                                        np.load(os.path.join(data_dir, 'weibo_id_712/weibo_test_fold0.npy'))
        weibo_ood_train, weibo_ood_valid, weibo_ood_test = np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_train_fold0.npy')), \
                                                           np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_valid_fold0.npy')), \
                                                           np.load(os.path.join(data_dir, 'weibo_ood_712/weibo_test_fold0.npy'))

        whole_dgl_dataset = bigcn_disc_dataset1(DATASET_NAME, data_dir, weibo_id_train, weibo_id_valid, weibo_id_test, \
                                                weibo_ood_train, weibo_ood_valid, weibo_ood_test)
        return whole_dgl_dataset

    print('Dataset not found')
    return None