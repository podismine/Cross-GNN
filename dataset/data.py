import numpy as np
from sklearn.utils import shuffle
import torch
from scipy.io import loadmat,savemat

def load_fold_data(args,seed = int(1111), device = 'gpu'):
    print("#" * 40)
    print(args)
    print("#" * 40)

    if args.dataset == "AD":
        all_data = loadmat("./dataset/AD.mat")
    elif args.dataset == "PD":
        all_data = loadmat("./dataset/PD.mat") #new2_NC_PD_10folds ppmi->xuanwu
    else:
        print(args.dataset)
        exit()

    # read data
    train_data = all_data[f'fold_{args.fold}']['train_data'][0,0]
    val_data = all_data[f'fold_{args.fold}']['val_data'][0,0]
    test_data = all_data[f'fold_{args.fold}']['test_data'][0,0]

    # read label
    train_label = all_data[f'fold_{args.fold}']['train_label'][0,0][:,0]
    val_label = all_data[f'fold_{args.fold}']['val_label'][0,0][:,0]
    test_label = all_data[f'fold_{args.fold}']['test_label'][0,0][:,0]

    # filter 3 classes for AD MCI NC
    use_index_train = [idx for idx,val in enumerate(train_label) if val != args.no]
    use_index_val = [idx for idx,val in enumerate(val_label) if val != args.no]
    use_index_test = [idx for idx,val in enumerate(test_label) if val != args.no]

    # get index
    print(f"Discarding... {args.no}")
    train_data = train_data[use_index_train]
    val_data = val_data[use_index_val]
    test_data = test_data[use_index_test]

    train_label=train_label[use_index_train]
    val_label=train_label[use_index_val]
    test_label=test_label[use_index_test]

    # map label
    max_label,min_label = max(train_label),min(train_label)

    train_label[train_label==min_label]=0
    train_label[train_label==max_label]=1

    val_label[val_label==min_label]=0
    val_label[val_label==max_label]=1

    test_label[test_label==min_label]=0
    test_label[test_label==max_label]=1

    # map nb of classes
    num_classes = len(np.unique(train_label))

    # make onehot labels
    val_onehot_labels = np.zeros((len(val_label), num_classes))
    test_onehot_labels = np.zeros((len(test_label), num_classes))

    for i in range(len(val_label)):
        val_onehot_labels[i, val_label[i]]=1

    for i in range(len(test_label)):
        test_onehot_labels[i, test_label[i]]=1

    # transform numpy to tensor
    train_data = torch.Tensor(train_data).float().cuda().permute(0,3,1,2) # N,2,M,M
    val_data = torch.Tensor(val_data).float().cuda().permute(0,3,1,2) # N,2,M,M
    test_data = torch.Tensor(test_data).float().cuda().permute(0,3,1,2) # N,2,M,M

    train_label = torch.Tensor(train_label).long().cuda()
    val_label = torch.Tensor(val_label).long().cuda()
    test_label = torch.Tensor(test_label).long().cuda()

    val_onehot_labels = torch.Tensor(val_onehot_labels).long().cuda()
    test_onehot_labels = torch.Tensor(test_onehot_labels).long().cuda()

    print("load data finished.")
    return train_data, val_data, test_data, train_label, val_label, test_label,val_onehot_labels, test_onehot_labels, num_classes