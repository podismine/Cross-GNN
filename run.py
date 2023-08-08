import torch
import numpy as np
import torch.optim as optim
import warnings
from args import get_parse
from dataset.data import load_fold_data
from models.cross import Cross
import random
import os
from train import train
warnings.filterwarnings("ignore")
seed = int(42)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

args = get_parse()

epochs = 400
env_name = f"test"
use_model = Cross
stop_epochs = 50

accs = []
aucs = []
sens = []
spes = []
f1s = []

early_stop = 0

corrs = []

for fold in range(10):
    print("Conducting... ",fold)
    args.fold=fold
    datas = load_fold_data(args, seed)

    in_channel = datas[0].size(1)
    kernel = datas[0].size(2)
    print(f"train_data shape {datas[0].shape}")
    print(f"val_data shape {datas[1].shape}")
    print(f"test_data shape {datas[2].shape}")

    model = use_model(in_channel=in_channel,kernel_size=kernel,num_classes = datas[-1],args=args)

    optimizer = optim.Adam(model.parameters(),
                        lr=3e-4, weight_decay=0.00005)

    model.cuda()

    res = {
    "best_acc": -99,
    "best_auc": 0.,
    "best_sen": 0.,
    "best_spe": 0.,
    "best_f1": 0.
    }

    for epoch in range(epochs):
        res,is_best, to_save = train(epoch, model,optimizer,datas ,res, args)
        if is_best is True:
            early_stop = 0
        else:
            continue
        if epoch > epochs//2:
            early_stop += 1
        if early_stop == stop_epochs:
            print("early stop. epoch: ", epoch)
            break
    accs.append(res['best_acc'])
    aucs.append(res['best_auc'])

    sens.append(res['best_sen'])
    spes.append(res['best_spe'])
    f1s.append(res['best_f1'])

print("accs: ",accs)
print("%.4f %.4f" % (np.mean(accs) * 100,np.std(accs) * 100) )
print("%.4f %.4f" % (np.mean(aucs) * 100,np.std(aucs) * 100) )
print("%.4f %.4f" % (np.mean(sens) * 100,np.std(sens) * 100) )
print("%.4f %.4f" % (np.mean(spes) * 100,np.std(spes) * 100) )
print("%.4f %.4f" % (np.mean(f1s) * 100,np.std(f1s) * 100) )