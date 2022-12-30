from random import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from umap import UMAP
# train/test split

cancer = load_breast_cancer()

train = pd.DataFrame(cancer.data, columns=cancer.feature_names)
train['target'] = cancer.target
# data['id'] = list(range(len(data)))

# train = data.sample(frac=0.9, random_state=2022).reset_index(drop=True)
# test = data[~data['id'].isin(train['id'])].reset_index(drop=True)

d = {}

for ndim in range(1, 31):
    # umap = UMAP(n_components=ndim, random_state=2022)
    # umap.fit(train[cancer.feature_names])

    # train_reduce = umap.transform(train[cancer.feature_names])
    # test_reduce = umap.transform(test[cancer.feature_names])

    # for i in range(ndim):
    #     train[f'col_{i}'] = train_reduce[:, i]
    #     test[f'col_{i}'] = test_reduce[:, i]
    res = UMAP(n_components=ndim, random_state=2022).fit_transform(train[cancer.feature_names])

    for i in range(ndim):
        train[f'col_{i}'] = res[:, i]

    features = [f'col_{i}' for i in range(ndim)]

    target = train['target']

    # k-fold on train_set
    nfold = 5

    oof_preds = np.zeros((train.shape[0]))
    oof_label = np.zeros(train.shape[0])
    # test_preds = pd.DataFrame({'id': test['id'], 'target': np.zeros(len(test))}, columns=['id', 'target'])

    folds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    for i, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
        print('---------- fold', i + 1, '----------')
        trn_X, val_X = train[features].iloc[trn_idx, :], train[features].iloc[val_idx, :]
        trn_y, val_y = target[trn_idx], target[val_idx]

        dtrn = lgb.Dataset(trn_X, label=trn_y)
        dval = lgb.Dataset(val_X, label=val_y)

        parameters = {
            'learning_rate': 0.1,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'nthread': 12
        }

        lgb_model = lgb.train(
            parameters,
            dtrn,
            num_boost_round=1000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=50,
        )
        oof_preds[val_idx] = lgb_model.predict(val_X[features], num_iteration=lgb_model.best_iteration)
        oof_label[val_idx] = val_y

        # test_preds['target'] += lgb_model.predict(test[features], num_iteration=lgb_model.best_iteration) / nfold
    # print(lgb_model.feature_importance(importance_type='gain'))

    print('----------- over -----------')
    oof_auc = roc_auc_score(oof_label, oof_preds)
    print('oof auc:', oof_auc)
    # print('test auc', roc_auc_score(test['target'], test_preds['target'])) 

    d[ndim] = oof_auc

print(d)
