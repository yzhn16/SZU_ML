from random import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_percentage_error, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
                                        
tqdm.pandas(desc='apply')

arr = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_arrythmia.txt', encoding='GBK',sep='\t', header=None)[0]
_dict = {k: v for v, k in enumerate(arr)}

names = ['key', 'age', 'gender'] + ['arr_{}'.format(i) for i in range(55)]
train_emb = pd.read_csv(f'./train_embeddings_SSL.csv')
train_info = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt', sep='\t', names=names, low_memory=False).fillna(-1)
test_emb = pd.read_csv(f'./test_embeddings_SSL.csv')
test_info = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_subA.txt', sep='\t', names=names, low_memory=False).fillna(-1)

def gen_onehot(x):
    for i in range(55):
        cur = x['arr_{}'.format(i)]
        if cur != -1:
            x['type_{}'.format(_dict[cur])] = 1
        else:
            break
    return x

def gen_features(df):
    arr = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_arrythmia.txt', encoding='GBK',sep='\t', header=None)[0]
    _dict = {k: v for v, k in enumerate(arr)}
    
    for i in range(55):
        df['type_{}'.format(i)] = 0
       
    df = df.progress_apply(lambda x: gen_onehot(x), axis=1)
    
    # df['age'] = df['age'].fillna(-1)
    df['gender'] = df['gender'].map({'FEMALE': 0, 'MALE': 1}) # .fillna(-1)

    return df


train = gen_features(train_info.merge(train_emb, on='key')).reset_index(drop=True)
test = gen_features(test_info.merge(test_emb, on='key')).reset_index(drop=True)


drop_features = ['key'] + ['arr_{}'.format(i) for i in range(55)] + ['type_{}'.format(i) for i in range(55)]
features = [c for c in train.columns if c not in drop_features]
print(features)
# k-fold on train_set
nfold = 5

sub = pd.DataFrame()
sub['key'] = test['key']
for t in tqdm(range(55)):
    print('-' * 20, 'task', t, '-' * 20)
    target = train['type_{}'.format(t)]

    oof_preds = np.zeros(train.shape[0])
    oof_label = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])

    folds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=3407)
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
            'metric': 'AUC',
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 32,
            'is_unbalance': True,
            'verbose': -1,
            'nthread': 12
        }

        lgb_model = lgb.train(
            parameters,
            dtrn,
            num_boost_round=3000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
    
        oof_preds[val_idx] = lgb_model.predict(val_X[features], num_iteration=lgb_model.best_iteration)
        oof_label[val_idx] = val_y
    
        test_preds += lgb_model.predict(test[features], num_iteration=lgb_model.best_iteration) / nfold
        
    print('oof auc:', roc_auc_score(oof_label, oof_preds))
    
    sub['type_{}'.format(t)] = test_preds
    sub.to_csv("submission_SL_GBDT.csv", index=False)
sub.to_csv("submission_SL_GBDT.csv", index=False)