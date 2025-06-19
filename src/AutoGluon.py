import os
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# ====================== 数据集路径 ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
train_path = os.path.join(project_dir, 'dataset', 'train.csv')
test_path = os.path.join(project_dir, 'dataset', 'test.csv')
output_path = os.path.join(project_dir, 'dataset', 'submission_AutoGluon.csv')

train_data = TabularDataset(train_path)
id, label = 'Id', 'Sold Price'
# 数据预处理
large_val_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount',
                  'Listed Price', 'Last Sold Price']
for c in large_val_cols + [label]:
    train_data[c] = np.log(train_data[c] + 1)

predictor = TabularPredictor(label=label).fit(
    train_data.drop(columns=[id])
)

# # 更好的模型
# predictor = TabularPredictor(label=label).fit(
#     train_data.drop(columns=[id]),
#     hyperparameters='multimodal',
#     num_stack_levels=1,
#     num_bag_folds=5
# )

# 预测
test_data = TabularDataset(test_path)
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id: test_data[id], label: preds})
submission.to_csv(output_path, index=False)
