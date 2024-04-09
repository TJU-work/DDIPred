from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve, precision_score, recall_score, auc
import sklearn.metrics as m
from utils import write_log
from sklearn.preprocessing import label_binarize

class Metric(Callback):
    def __init__(self, x_train, y_train, x_valid, y_valid,aggregator_type,dataset,K_fold, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.aggregator_type=aggregator_type
        self.dataset=dataset
        self.k=K_fold
        self.threshold=0.5
        self.batch_size = batch_size


        super(Metric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_valid, batch_size=self.batch_size)
        y_true = self.y_valid

        y_true_labels = np.zeros(len(y_true,),dtype=int)
        for i,out in enumerate(y_true):
            y_true_labels[i] = np.argmax(out)

        y_pred_labels = np.zeros(len(y_pred,),dtype=int)
        for i,out in enumerate(y_pred):
            y_pred_labels[i] = np.argmax(out)
        try:
            auc = roc_auc_score(y_true=y_true_labels, y_score=y_pred, average='macro')
        except:
            auc = 0
        try:
            aupr = self.roc_aupr_score(y_true, y_pred, average='macro')
        except:
            aupr = 0

        acc = accuracy_score(y_true=y_true_labels, y_pred=y_pred_labels)
        f1_macro = f1_score(y_true=y_true_labels, y_pred=y_pred_labels, average='macro')
        recall_macro = recall_score(y_true=y_true_labels, y_pred=y_pred_labels,average='macro')
        precision_macro = precision_score(y_true=y_true_labels, y_pred=y_pred_labels,average='macro')

        logs['val_aupr'] = float(aupr)
        logs['val_auc'] = float(auc)
        logs['val_acc'] = float(acc)
        logs['val_f1_macro'] = float(f1_macro)
        logs['val_rec_macro'] = float(recall_macro)
        logs['val_pre_macro'] = float(precision_macro)
        
        logs['dataset'] = self.dataset
        logs['aggregator_type'] = self.aggregator_type
        logs['kfold'] = self.k
        logs['epoch_count'] = epoch+1
        print(f'Logging Info - epoch: {epoch+1}, macro_auc: {auc}, macro_aupr: {aupr}, acc: {acc}, macro_f1: {f1_macro}, \
                macro_rec: {recall_macro}, macro_prec: {precision_macro}')
        write_log('log/train_history.txt',logs,mode='a')


    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict


    def roc_aupr_score(self, y_true, y_score, average):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
            return m.auc(recall, precision)

        def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
            if average == "binary":
                return binary_metric(y_true, y_score)
            if average == "micro":
                y_true = y_true.ravel()
                y_score = y_score.ravel()
            if y_true.ndim == 1:
                y_true = y_true.reshape((-1, 1))
            if y_score.ndim == 1:
                y_score = y_score.reshape((-1, 1))
            n_classes = y_score.shape[1]
            score = np.zeros((n_classes,))
            for c in range(n_classes):
                y_true_c = y_true.take([c], axis=1).ravel()
                y_score_c = y_score.take([c], axis=1).ravel()
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)
        return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def eval_ddi_types(true_labels, pred_labels):

    true_labels_uniq = np.unique(true_labels)
    ddi_types = len(true_labels_uniq)
    one_hot_true = label_binarize(true_labels, classes=true_labels_uniq)
    one_hot_pred = label_binarize(pred_labels, classes=true_labels_uniq)
    result_eve = []

    for i in range(ddi_types):
        metrics = {"label":0,"ACC":0, "AUPR": 0, "AUC": 0, "F1":0}
        metrics["label"] = true_labels_uniq[i]
        metrics["ACC"] = accuracy_score(one_hot_true.take([i], axis=1).ravel(), one_hot_pred.take([i], axis=1).ravel())
        metrics["AUPR"] = roc_aupr_score(one_hot_true.take([i], axis=1).ravel(), one_hot_pred.take([i], axis=1).ravel(),
                                          average=None)
        metrics["AUC"] = roc_auc_score(one_hot_true.take([i], axis=1).ravel(), one_hot_pred.take([i], axis=1).ravel(),
                                         average=None)
        metrics["F1"] = f1_score(one_hot_true.take([i], axis=1).ravel(), one_hot_pred.take([i], axis=1).ravel(),
                                    average='binary')

        result_eve.append(metrics)
    return result_eve
