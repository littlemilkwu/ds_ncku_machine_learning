import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report


def train_test_split(df, y=None, random_state=None, train_size=0.7):
    index = np.array(range(len(df)))
    if random_state is not None:
        np.random.RandomState(random_state).shuffle(index)
    else:
        np.random.shuffle(index)
    train_index = index[:round(len(df) * train_size)]
    test_index = index[round(len(df) * train_size):]
    # return(train_index[:10])
    if y is not None:
        return (df.loc[train_index].reset_index(drop=True), df.loc[test_index].reset_index(drop=True),
                y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True))
    return (df.loc[train_index].reset_index(drop=True), df.loc[test_index].reset_index(drop=True))

def cross_validation(X:np.ndarray, y:np.ndarray, k=5, random_state=None):
    index = [range(len(X))]
    cv_len = len(X) // k
    if random_state is None:
        np.random.shuffle(index)
    else:
        np.random.RandomState(random_state).shuffle(index)

    start = 0
    for i in range(k):
        start = i * cv_len
        end = start + cv_len
        train_X = np.concatenate([X[:start], X[end:]], axis=0)
        train_y = np.concatenate([y[:start], y[end:]], axis=0)
        val_X = X[start:end]
        val_y = y[start:end]
        yield train_X, val_X, train_y, val_y, i

def one_hot_encode(df, cols):
    for col in cols:
        unique_value = df[col].unique()
        for val in unique_value:
            new_col_name = f"{col}_{val}"
            df[new_col_name] = 0
            mask = (df[col] == val)
            df.loc[mask, new_col_name] = 1
        
    df = df.drop(columns=cols)
    return df


def target_encode(df, y, col):
    dict_map = dict()
    unique_value = df[col].unique()
    for val in unique_value:
        mask = (df[col] == val)
        mask = mask[mask].index.tolist()
        mean_of_y = (y[mask].sum()) / len(y[mask])
        dict_map[val] = mean_of_y
    df[col] = df[col].replace(dict_map)
    return df, dict_map


def yes_no_01(df, cols):
    for col in cols:
        df[col] = df[col].replace({"Yes": 1, "No": 0}).astype(int)
    return df

def evaluate(true_y, pred_y):
    print('{:<15} :'.format('Accuracy'), accuracy_score(true_y, pred_y))
    print('{:<15} :'.format('Precision'), precision_score(true_y, pred_y))
    print('{:<15} :'.format('Recall'), recall_score(true_y, pred_y))
    print('{:<15} :'.format('F1'), f1_score(true_y, pred_y))
    print(classification_report(true_y, pred_y))