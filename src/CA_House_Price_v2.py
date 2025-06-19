import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 数据集路径 ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
train_path = os.path.join(project_dir, 'dataset', 'train.csv')
test_path = os.path.join(project_dir, 'dataset', 'test.csv')
output_path = os.path.join(project_dir, 'dataset', 'submission_v2.csv')
imgs_path = os.path.join(project_dir, 'imgs')
pretrain_model_path = os.path.join(project_dir, 'pretrained', 'pretrained_house_model.pth')
pretrain_scaler_path = os.path.join(project_dir, 'pretrained', 'pretrained_scaler.pkl')

# 确保目录存在
os.makedirs(os.path.dirname(pretrain_model_path), exist_ok=True)
os.makedirs(os.path.dirname(pretrain_scaler_path), exist_ok=True)
os.makedirs(imgs_path, exist_ok=True)


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
    special_columns = ['Bedrooms', 'Parking', 'Total interior livable area', 'Bathrooms', 'Lot', 'Year built']

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

    # ====================== 特征工程增强 ======================
    # 添加新特征
    if 'Sold Price' in all_features.columns and 'Total interior livable area' in all_features.columns:
        all_features['PricePerSqft'] = all_features['Sold Price'] / (all_features['Total interior livable area'] + 1e-6)
    if 'Bedrooms' in all_features.columns and 'Bathrooms' in all_features.columns:
        all_features['RoomRatio'] = all_features['Bedrooms'] / (all_features['Bathrooms'] + 1e-6)
    if 'Year built' in all_features.columns:
        current_year = pd.Timestamp.now().year
        all_features['HouseAge'] = current_year - all_features['Year built']

    # 交互特征
    for col1 in ['Total interior livable area', 'Lot']:
        for col2 in ['Bedrooms', 'Bathrooms']:
            if col1 in all_features.columns and col2 in all_features.columns:
                all_features[f'{col1}_{col2}'] = all_features[col1] * all_features[col2]

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

    # 标签标准化 - 使用分位数归一化
    label_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    train_labels = label_scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()

    # 重新分割数据集
    n_train = len(train_data)
    processed_train = all_features.iloc[:n_train]
    processed_test = all_features.iloc[n_train:]

    print(f"特征维度: {processed_train.shape[1]}")

    return processed_train, train_labels, processed_test, test_ids, label_scaler, scaler


# ====================== 神经网络模型 ======================
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.res_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        residual = self.res_layer(x)
        x = x + residual  # 残差连接
        return self.output(x)


# ====================== 训练函数 ======================
def train_model(model, train_loader, val_loader, epochs=200, lr=0.0005, freeze_layers=False):
    if freeze_layers:
        # 冻结前两层
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        print("冻结前两层参数，只训练后续层")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // 2)
    criterion = nn.SmoothL1Loss()

    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model = None
    patience, counter = 15, 0

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

        # 更新学习率
        scheduler.step()

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

        # 早停检查和模型保存
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停于第 {epoch + 1} 轮")
                break

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}')

    # 加载最佳模型
    model.load_state_dict(best_model)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(imgs_path, 'loss_curve_v2.png'))
    plt.close()

    return model


# ====================== 预训练函数 ======================
def pretrain_model():
    print("开始预训练模型...")
    # 加载并预处理数据
    train_features, train_labels, _, _, _, scaler = load_and_preprocess_data()

    # 保存预训练使用的scaler
    joblib.dump(scaler, pretrain_scaler_path)
    print(f"保存预训练scaler到: {pretrain_scaler_path}")

    # 转换为PyTorch张量
    x_train = torch.tensor(train_features.values.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(train_labels.astype(np.float32), dtype=torch.float32)

    # 创建完整训练集数据集
    full_train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)

    # 初始化模型
    input_size = x_train.shape[1]
    model = HousePricePredictor(input_size).to(device)
    print(f"预训练模型初始化完成，输入特征维度: {input_size}")

    # 训练预训练模型
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.SmoothL1Loss()

    best_loss = float('inf')
    epochs = 100

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), pretrain_model_path)

        if (epoch + 1) % 20 == 0:
            print(f'预训练 Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}')

    print(f"预训练完成，模型保存到: {pretrain_model_path}")
    return model


# ====================== 模型集成预测 ======================
def ensemble_predict(models, test_tensor):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(test_tensor).cpu().numpy()
            predictions.append(pred)
    return np.mean(predictions, axis=0)


# ====================== 主程序 ======================
def main():
    # 加载并预处理数据
    print("加载并预处理数据...")
    train_features, train_labels, test_features, test_ids, label_scaler, scaler = load_and_preprocess_data()

    # 检查并加载预训练模型
    if not os.path.exists(pretrain_model_path):
        print("未找到预训练模型，开始预训练...")
        pretrain_model()

    print("加载预训练模型...")
    # 转换为PyTorch张量
    x_train = torch.tensor(train_features.values.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(train_labels.astype(np.float32), dtype=torch.float32)

    # 初始化模型
    input_size = x_train.shape[1]
    pretrained_model = HousePricePredictor(input_size).to(device)
    pretrained_model.load_state_dict(torch.load(pretrain_model_path))
    print(f"加载预训练模型完成，输入特征维度: {input_size}")

    # 创建完整训练集数据集
    full_train_dataset = TensorDataset(x_train, y_train)

    # 使用PyTorch的random_split
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

    # 训练多个模型进行集成
    num_models = 5
    models = []

    for i in range(num_models):
        print(f"\n训练模型 {i + 1}/{num_models}")
        # 每次使用不同的随机种子初始化
        torch.manual_seed(42 + i)

        # 创建新模型实例并加载预训练权重
        model = HousePricePredictor(input_size).to(device)
        model.load_state_dict(torch.load(pretrain_model_path))

        # 微调模型（冻结前两层）
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=100,
            lr=0.0003,
            freeze_layers=True
        )
        models.append(trained_model)
        print(f"模型 {i + 1} 训练完成")

    # 在测试集上进行预测
    print("在测试集上进行集成预测...")
    test_tensor = torch.tensor(test_features.values.astype(np.float32), dtype=torch.float32).to(device)

    # 使用模型集成进行预测
    predictions = ensemble_predict(models, test_tensor)

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