# 机器学习课程作业@SZU
## Ex1:威斯康星州乳腺癌诊断情况预测
### 环境依赖
- Python 3.9.12
- numpy == 1.21.5
- pandas == 1.4.2
- lightgbm == 3.3.2
- scikit-learn == 1.0.2
### 目录结构
```
./Ex1
├── data.csv, csv格式的训练数据
├── run_lgb.py, 训练梯度提升决策树模型
├── run_lr.py, 训练逻辑回归模型
├── run_rf.py, 训练随机森林模型
├── run_svm.py, 训练支持向量分类模型
└── ...
```
### 运行流程
```
cd ./Ex1
python run_[MASK].py
```

## Ex2: 数据的低维特征提取
### 环境依赖
- Python 3.9.12
- numpy == 1.21.5
- pandas == 1.4.2
- lightgbm == 3.3.2
- scikit-learn == 1.0.2
- umap-learn == 0.5.3
### 目录结构
```
./Ex2
├── reduce.py, 降维分析
├── train.py, 使用不同的降维维度训练分类模型
└── ...
```

## Ex3:心电异常事件预测
- 2019 “合肥高新杯” 心电人机智能
### 环境依赖
- Python 3.9.12
- numpy == 1.21.5
- pandas == 1.4.2
- lightgbm == 3.3.2
- scikit-learn == 1.0.2
- torch == 1.12.1
