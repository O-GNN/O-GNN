import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def eval_acc(y_true, y_pred):
    y_pred = (y_pred > 0).astype(np.float32)
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == y_true[:, i]
        if not np.any(is_labeld):
            continue
        correct = y_true[is_labeld, i] == y_pred[is_labeld, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    rocauc_list = []
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == y_true[:, i]
        if not np.any(is_labeld):
            continue
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true[is_labeld, i], y_pred[is_labeld, i]))

    return sum(rocauc_list) / len(rocauc_list)


def eval_posacc(y_true, y_pred):
    y_pred = (y_pred > 0).astype(np.float32)
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == 1
        if not np.any(is_labeld):
            continue
        correct = y_pred[is_labeld, i] == 1
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def eval_negacc(y_true, y_pred):
    y_pred = (y_pred > 0).astype(np.float32)
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == 0
        if not np.any(is_labeld):
            continue
        correct = y_pred[is_labeld, i] == 0
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def eval_ap(y_true, y_pred):
    ap_list = []
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == y_true[:, i]
        if not np.any(is_labeld):
            continue
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            ap_list.append(average_precision_score(y_true[is_labeld, i], y_pred[is_labeld, i]))
    return sum(ap_list) / len(ap_list)


def eval_f1(y_true, y_pred):
    f1_list = []
    y_pred = (y_pred > 0).astype(np.float32)
    for i in range(y_true.shape[1]):
        is_labeld = y_true[:, i] == y_true[:, i]
        if not np.any(is_labeld):
            continue
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            f1_list.append(f1_score(y_true[is_labeld, i], y_pred[is_labeld, i]))

    return sum(f1_list) / len(f1_list)


metric2fun = {
    "acc": eval_acc,
    "rocauc": eval_rocauc,
    "posacc": eval_posacc,
    "negacc": eval_negacc,
    "ap": eval_ap,
    "f1": eval_f1,
}


class Evaluator:
    def __init__(self, dataset=None):
        self.dataset = dataset

    @property
    def evaluation_metrics(self):
        return ["rocauc", "acc", "posacc", "negacc", "ap", "f1"]

    def evaluate(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        output_dict = dict()
        metrics = self.evaluation_metrics

        for metric in metrics:
            output_dict[metric] = metric2fun[metric](y_true, y_pred)
        return output_dict
