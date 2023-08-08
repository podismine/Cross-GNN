import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import torch
from termcolor import cprint
from sklearn.utils import shuffle

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


best_val_acc = 0.

def train(epoch, model,optimizer, datas, res, args):
    global best_val_acc
    train_data, val_data, test_data, \
        train_label, val_label, test_label, \
            val_onehot_labels, test_onehot_labels, _ = datas
    is_best = False
    t = time.time()
    model.train()

    batch_size = 64
    batch_iters = len(train_data) // batch_size + 1
    acc_train_loss = 0.
    cum_acc_train = 0.
    train_data, train_label = shuffle(train_data, train_label)
    for i in range(batch_iters):
        st = i * batch_size
        en = min(st + batch_size, len(train_data))
        if st == en or st == en -1:
            continue
        sub_train_x = train_data[st:en]
        sub_train_y = train_label[st:en]

        alpha_t = args.alpha * ((epoch + 1) / 50)

        optimizer.zero_grad()
        output1, p1, p2 = model(sub_train_x, tem = 1)
        loss_train = F.cross_entropy(output1, sub_train_y)  \
            + alpha_t * F.kl_div(p1, output1) + F.kl_div(p2, output1) * alpha_t + F.kl_div(p2, p1.exp()) * alpha_t

        acc_train = accuracy( output1, sub_train_y)
        loss_train.backward()
        optimizer.step()
        acc_train_loss += float(loss_train) * len(sub_train_x)
        cum_acc_train += float(acc_train) * len(sub_train_x)
    acc_train_loss /= len(train_data)
    cum_acc_train /= len(train_data)
    
    model.eval()
    with torch.no_grad():
        pred_val, _, _ = model(val_data, get_corr = False)

    acc_val = accuracy(pred_val,val_label).cpu().detach().numpy()
    to_save = {}
    if best_val_acc <= acc_val:
        is_best = True
        best_val_acc = acc_val

        with torch.no_grad():
            pred_test, _, _ = model(test_data, get_corr = False)
        acc_test = accuracy(pred_test,test_label).cpu().detach().numpy()

        onehot_test_label_cpu = test_onehot_labels.cpu().detach().numpy()
        pred_test_cpu = pred_test.cpu().detach().numpy()
        pred_label_cpu = pred_test.max(1)[1].cpu().detach().numpy()
        test_label_cpu = test_label.cpu().detach().numpy()
        auc_test = roc_auc_score(onehot_test_label_cpu.ravel(), pred_test_cpu.ravel())
        to_save = {
            #"corr": corr.cpu().detach().numpy(),
            "label_onehot": onehot_test_label_cpu,
            "label": test_label_cpu
        }
        cm = confusion_matrix(pred_label_cpu,test_label_cpu)
        f1 = f1_score(pred_label_cpu,test_label_cpu)
        eval_sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
        eval_spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)
        res['best_acc']  = acc_test
        res['best_auc']  = auc_test
        res['best_sen']  = eval_sen
        res['best_spe']  = eval_spe
        res['best_f1']  = f1
        cprint('Epoch: {:04d} '.format(epoch+1) +
            'loss_train: {:.4f} '.format(loss_train.item())+
            'acc_train: {:.4f} '.format(cum_acc_train)+
            'val_acc: {:.4f} '.format(acc_val)+
            'test_acc: {:.4f} '.format(acc_test)+
            'test_auc: {:.4f} '.format(auc_test)+
            'test_sen: {:.4f} '.format(eval_sen)+
            'test_spe: {:.4f} '.format(eval_spe)+
            'test_f1: {:.4f} '.format(f1)+
            'time: {:.4f}s '.format(time.time() - t),'green')
    return res, is_best, to_save