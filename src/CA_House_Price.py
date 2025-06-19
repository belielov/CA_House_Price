import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 数据集路径 ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
train_path = os.path.join(project_dir, 'dataset', 'train.csv')
test_path = os.path.join(project_dir, 'dataset', 'test.csv')
output_path = os.path.join(project_dir, 'dataset', 'submission.csv')
imgs_path = os.path.join(project_dir, 'imgs')


# ====================== 数据预处理 ======================
def load_and_preprocess_data():
    # 加载数据集
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 移除不相关特征
    irrelevant_columns = ['Address', 'Summary', 'State', 'Heating', 'Cooling', 'Elementary School',
                          'Middle School', 'High School', 'Flooring', 'Cooling features', 'Parking features',
                          'Appliances included', 'Laundry features', 'Last Sold On', 'Listed On', 'Id']

    # 需要特殊处理的列
    special_columns = ['Bedrooms', 'Parking']

    # 稀疏类别标签
    sparse_columns = ['Type', 'Region', 'Heating features', 'City']

    # 处理训练集
    train_features = train_data.drop(columns=['Sold Price'] + irrelevant_columns)
    train_labels = np.log1p(train_data['Sold Price'].values)  # 对数变换压缩范围

    # 处理测试集
    test_features = test_data.drop(columns=irrelevant_columns)
    test_ids = test_data['Id'].values

    # 合并数据集进行预处理
    all_features = pd.concat([train_features, test_features], axis=0)

    # 特殊列处理函数
    def process_special_columns(df, columns):
        for col in columns:
            # 处理非数值数据
            df[col] = df[col].apply(
                lambda x: str(x).count(',') + 1 if not str(x).replace('.', '').isdigit() else x
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 修复: 避免链式赋值
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        return df

    # 稀疏类别处理函数
    def process_sparse_columns(df, columns, threshold=1.0):
        for col in columns:
            # 计算类别频率
            value_counts = df[col].value_counts(normalize=True) * 100
            # 识别稀疏类别
            rare_categories = value_counts[value_counts < threshold].index
            # 替换为'Other'
            df[col] = df[col].apply(lambda x: 'Other' if x in rare_categories else x)
        return df

    # 应用处理
    all_features = process_special_columns(all_features, special_columns)
    all_features = process_sparse_columns(all_features, sparse_columns)

    # 分离数值特征和分类特征
    numeric_cols = all_features.select_dtypes(include=['number']).columns
    category_cols = all_features.select_dtypes(exclude=['number']).columns

    # 数值特征处理
    scaler = StandardScaler()
    # 修复: 避免链式赋值
    numeric_data = all_features[numeric_cols].fillna(all_features[numeric_cols].median())
    all_features[numeric_cols] = scaler.fit_transform(numeric_data)

    # 分类特征处理
    all_features = pd.get_dummies(all_features, columns=category_cols, dummy_na=True)

    # 标签标准化
    label_scaler = StandardScaler()
    train_labels = label_scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()

    # 重新分割数据集
    n_train = len(train_data)
    processed_train = all_features.iloc[:n_train]
    processed_test = all_features.iloc[n_train:]

    print(f"特征维度: {processed_train.shape[1]}")

    return processed_train, train_labels, processed_test, test_ids, label_scaler


# ====================== 神经网络模型 ======================
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


# ====================== 训练函数 ======================
def train_model(model, train_loader, val_loader, epochs=300, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    criterion = nn.SmoothL1Loss()  # 改用SmoothL1Loss对异常值更鲁棒

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(imgs_path, 'loss_curve.png'))
    plt.show()

    return model


# ====================== 主程序 ======================
def main():
    # 加载并预处理数据
    print("加载并预处理数据...")
    train_features, train_labels, test_features, test_ids, label_scaler = load_and_preprocess_data()

    # 转换为PyTorch张量
    x_train = torch.tensor(train_features.values.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(train_labels.astype(np.float32), dtype=torch.float32)

    # 创建完整训练集数据集
    full_train_dataset = TensorDataset(x_train, y_train)

    # 使用PyTorch的random_split替代sklearn的train_test_split
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    input_size = x_train.shape[1]
    model = HousePricePredictor(input_size).to(device)
    print(f"模型初始化完成，输入特征维度: {input_size}")

    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=15,  # 减少epoch数量
        lr=0.001
    )

    # 在测试集上进行预测
    print("在测试集上进行预测...")
    test_tensor = torch.tensor(test_features.values.astype(np.float32), dtype=torch.float32).to(device)
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(test_tensor).cpu().numpy()

    # 反向转换预测结果
    predictions = label_scaler.inverse_transform(predictions)
    predictions = np.expm1(predictions)  # 指数变换恢复原始范围

    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'Sold Price': predictions.flatten()
    })
    submission.to_csv(output_path, index=False)
    print(f"提交文件已保存为 '{output_path}'")


if __name__ == "__main__":
    main()
