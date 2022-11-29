import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

arr = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_arrythmia.txt', encoding='GBK',sep='\t', header=None)[0]
_dict = {k: v for v, k in enumerate(arr)}

submission = pd.read_csv('./submission_SSL_GBDT.csv')
names = ['key', 'age', 'gender'] + ['arr_{}'.format(i) for i in range(55)]
test_info = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/heifei_round1_ansA_20191008.txt', sep='\t', names=names, low_memory=False).fillna(-1)


for i in range(55):
    test_info['type_{}'.format(i)] = 0
    
def gen_onehot(x):
    for i in range(55):
        cur = x['arr_{}'.format(i)]
        if cur != -1:
            x['type_{}'.format(_dict[cur])] = 1
        else:
            break
    return x

test_info = test_info.apply(lambda x: gen_onehot(x), axis=1)

labels = ['type_{}'.format(i) for i in range(55)]
print(roc_auc_score(np.array(test_info[labels]), np.array(submission[labels]), average='macro'))


def f1(y_true, y_pred, threshold=0.5):
    y_true = y_true.flatten()
    y_pred = np.where(y_pred > threshold, 1, 0).flatten()

    return f1_score(y_true, y_pred)
print(f1(np.array(test_info[labels]), np.array(submission[labels])))

# SL + SSL
