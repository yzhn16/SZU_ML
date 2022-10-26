from random import random
import numpy as np
import pandas as pd
# import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# train/test split
data = pd.read_csv(f'data.csv')
train_data = data.sample(frac=0.8, random_state=42)
test_data = data[~data['id'].isin(train_data['id'])]

def gen_features(df):
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(np.int8)
    
    return df

train = gen_features(train_data).reset_index(drop=True)
test = gen_features(test_data).reset_index(drop=True)

drop_features = ['id', 'diagnosis', 'Unnamed: 32']
features = [c for c in train.columns if c not in drop_features]
target = train['diagnosis']

# k-fold on train_set
nfold = 5

oof_preds = np.zeros((train.shape[0]))
oof_label = np.zeros(train.shape[0])
test_preds = pd.DataFrame({'id': test['id'], 'diagnosis': np.zeros(len(test))}, columns=['id', 'diagnosis'])
feature_importance_df = pd.DataFrame()

folds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
for i, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
    print('---------- fold', i + 1, '----------')
    trn_X, val_X = train[features].iloc[trn_idx, :], train[features].iloc[val_idx, :]
    trn_y, val_y = target[trn_idx], target[val_idx]

    model = RandomForestClassifier(random_state=42)

    model.fit(trn_X, trn_y.values.ravel())

    oof_preds[val_idx] = model.predict_proba(val_X[features])[:,1]
    oof_label[val_idx] = val_y

    test_preds['diagnosis'] += model.predict_proba(test[features])[:,1] / nfold
 
 # *在训练集上找F1分数最优阈值
threshold = 0.5
max_f1 = 0
for t in range(1, 100):
    cur_f1 = f1_score(oof_label, np.where(oof_preds < (t / 100), 0, 1))
    if  cur_f1 > max_f1:
        max_f1 = cur_f1
        threshold = t / 100
test_preds['_diagnosis'] = test_preds['diagnosis'].apply(lambda x: 1 if x >= threshold else 0)

# eval on test_set
print('----------- over -----------')
print('oof auc:', roc_auc_score(oof_label, oof_preds))

print('test auc:', roc_auc_score(test['diagnosis'], test_preds['diagnosis']))
print('test acc:', accuracy_score(test['diagnosis'], test_preds['_diagnosis']))
print('test recall:', recall_score(test['diagnosis'], test_preds['_diagnosis']))
print('test precison:', precision_score(test['diagnosis'], test_preds['_diagnosis']))
print('test f1:', f1_score(test['diagnosis'], test_preds['_diagnosis']))