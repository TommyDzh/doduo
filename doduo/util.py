import random

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch

import wandb
from typing import Optional, Sequence
def log_confusion_matrix(conf_mat, class_names: Sequence = None , title: str = None, split_table: bool = False):
    assert len(conf_mat) == len(class_names), "Higher predicted index than number of classes"
    assert len(conf_mat) == len(conf_mat[0]), "Inequal number of rows and columns"
    n_classes = len(class_names)
    class_inds = [i for i in range(n_classes)]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(1, n_classes + 1)]

    # get mapping of inds to class index in case user has weird prediction indices
    class_mapping = {}
    for i, val in enumerate(sorted(list(class_inds))):
        class_mapping[val] = i
    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], conf_mat[i][j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    title = title or ""
    table = wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data)
    wandb.log({title: table})
    # wandb.log({f"{title}_plot":
    # wandb.plot_table(
    #     "wandb/confusion_matrix/v1",
    #     table,
    #     fields,
    #     {"title": title},
    #     split_table=split_table,
    # )})

def log_class_f1(ts_class_f1, class_names: Sequence = None , title: str = None, split_table: bool = False):
    assert len(ts_class_f1) == len(class_names), "Higher predicted index than number of classes"
    n_classes = len(class_names)
    class_inds = [i for i in range(n_classes)]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(1, n_classes + 1)]

    # get mapping of inds to class index in case user has weird prediction indices
    class_mapping = {}
    for i, val in enumerate(sorted(list(class_inds))):
        class_mapping[val] = i
    data = []
    for i in range(n_classes):
        data.append([class_names[i],ts_class_f1[i]])
            
    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    title = title or ""
    table = wandb.Table(columns=["Class", "F1"], data=data)
    wandb.log({title: table})

def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)


def parse_tagname(tag_name):
    """sato_bert_bert-base-uncased-bs16-ml-256"""
    if "__" in tag_name:
        # Removetraining ratio
        tag_name = tag_name.split("__")[0]
    tokens = tag_name.split("_")[-1].split("-")
    shortcut_name = "-".join(tokens[:-3])
    max_length = int(tokens[-1])
    batch_size = int(tokens[-3].replace("bs", ""))
    return shortcut_name, batch_size, max_length


def set_seed(seed: int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """
