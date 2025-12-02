import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

import os
import joblib
import numpy as np
import pandas as pd
import holidays
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import random
# 从 lstm.py 中导入纯粹的模型类
from models.model_lstm_attention import TlcModel
model_name = "lstm_attention"
# --- 自定义评估指标函数 (已移动到这里) ---
def concordance_correlation_coefficient(y_true, y_pred):
    """计算一致性相关系数"""
    # 展平数组以进行整体计算
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.cov(y_true, y_pred)[0][1]
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)

def pearson_r(y_true, y_pred):
    """计算皮尔逊相关系数"""
    # 展平数组以进行整体计算
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return pearsonr(y_true, y_pred)[0]

# --- 数据处理和特征工程函数 ---
def load_and_preprocess_data(data_file):
    """
    加载数据并进行预处理和特征工程。
    返回包含所有特征的DataFrame和目标特征维度。
    """
    try:
        df = pd.read_csv(data_file, parse_dates=[0])
        # 获取目标特征的维度（第一列是时间戳，不计入）
        target_dim = df.shape[1] - 1

        # 创建用于模型的DataFrame, 包含所有原始特征
        df_for_model = df.iloc[:, 1:].copy()

        # 添加星期和小时列
        df['weekday'] = df.iloc[:, 0].dt.dayofweek
        df['hour'] = df.iloc[:, 0].dt.hour

        # 新增人工创建的特征
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] < 10)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] < 20)).astype(int)
        us_holidays = holidays.country_holidays('US', years=[2025])
        df['is_holiday'] = df.iloc[:, 0].dt.date.isin(us_holidays).astype(int)

        target_features = df_for_model.columns[:target_dim]
        other_features = ['weekday', 'hour', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_holiday']

        df_target = df_for_model[target_features].copy()
        df_other = df[other_features].copy()

        # 确保列名是字符串类型
        df_target.columns = df_target.columns.astype(str)
        df_other.columns = df_other.columns.astype(str)

        print("\n--- 送入模型的**人工创建**特征列表 ---")
        print(other_features)
        print("----------------------------")
        print("数据加载与特征创建成功，目标维度:", target_dim)

        return df_target, df_other, target_dim
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：找不到文件 '{data_file}'。请确保文件路径正确。")

def create_sequences(data, seq_length, target_dim):
    """
    创建重叠的序列数据。
    输入X包含所有特征，输出y只包含目标特征。
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, :target_dim]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def continuous_split(data, split_ratio):
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data

# --- 主程序流程 ---
if __name__ == "__main__":
    # --- 新增: 设置随机种子以保证可复现性 ---
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 检查GPU设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 定义文件路径和超参数 ---
    data_file = './datasets/tripdata.csv'

    # 根据模型名称创建动态路径
    model_dir = os.path.join('./models', model_name)
    report_dir = os.path.join('./reports', model_name)
    SEQ_LENGTH = 10
    SPLIT_RATIO = 0.8
    EPOCHS = 100
    BATCH_SIZE = 32
    lr=0.001

    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 1. 数据加载与预处理
    try:
        df_target, df_other, TARGET_DIM = load_and_preprocess_data(data_file)
    except FileNotFoundError as e:
        print(e)
        exit()

    # 对原始目标特征进行归一化
    target_scaler = MinMaxScaler()
    scaled_target_data = target_scaler.fit_transform(df_target.values)
    joblib.dump(target_scaler, os.path.join(model_dir, 'target_minmax_scaler.pkl'))
    print(f"目标特征 MinMaxScaler 已保存到 '{model_dir}' 目录。")

    # 拼接归一化后的目标特征和未缩放的其他特征
    scaled_data = np.hstack((scaled_target_data, df_other.values))

    # 2. 创建连续的序列数据
    X, y = create_sequences(scaled_data, SEQ_LENGTH, TARGET_DIM)

    # 划分训练集和验证集
    X_train_np, X_val_np = continuous_split(X, SPLIT_RATIO)
    y_train_np, y_val_np = continuous_split(y, SPLIT_RATIO)

    print(f"训练集形状: X={X_train_np.shape}, y={y_train_np.shape}")
    print(f"验证集形状: X={X_val_np.shape}, y={y_val_np.shape}")

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).to(device)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 构建 LSTM 模型
    INPUT_DIM = X.shape[2]
    OUTPUT_DIM = y.shape[1]
    model = TlcModel(input_dim=INPUT_DIM, hidden_dim=64, output_dim=OUTPUT_DIM, max_seq_len=SEQ_LENGTH,device=device).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 将用于保存最佳模型的指标从 val_loss 切换到 R2
    # --- 修改: 切换到用 MAE 作为保存模型的指标，因此初始化为正无穷 ---
    best_val_mae = float('inf')
    best_model_path = os.path.join(model_dir, f'best_lstm_model_{model_name}.pth')

    # 新增: 用于保存最佳性能指标的字典
    best_val_metrics = {}

    train_losses_history = []
    val_losses_history = []
    val_rmses = []
    val_maes = []
    val_mses = []
    val_rs = []
    val_r2s = []
    val_cccs = []

    print("\n开始模型训练...")
    # 4. 训练模型并评估
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = train_loss / len(train_dataset)
        train_losses_history.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_pred = model(X_val_batch)
                val_loss += criterion(val_pred, y_val_batch).item()
                val_preds_list.append(val_pred.cpu().numpy())
                val_targets_list.append(y_val_batch.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_losses_history.append(avg_val_loss)

            val_preds_np = np.concatenate(val_preds_list)
            val_targets_np = np.concatenate(val_targets_list)

            # 反归一化并处理负值
            val_pred_unscaled = target_scaler.inverse_transform(val_preds_np)
            val_y_unscaled = target_scaler.inverse_transform(val_targets_np)
            val_pred_unscaled[val_pred_unscaled < 0] = 0

            # 计算指标并存储
            y_true_flat = val_y_unscaled.flatten()
            y_pred_flat = val_pred_unscaled.flatten()

            rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
            mse = mean_squared_error(y_true_flat, y_pred_flat)
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            r = pearson_r(y_true_flat, y_pred_flat)
            r2 = r2_score(y_true_flat, y_pred_flat)
            ccc = concordance_correlation_coefficient(y_true_flat, y_pred_flat)

            # --- 修改保存模型的逻辑：根据 MAE 分数来判断是否保存 ---
            # MAE 越小越好，因此使用 '<'
            if mae < best_val_mae:
                best_val_mae = mae
                torch.save(model.state_dict(), best_model_path)
                print(f"新最佳模型已保存，验证 MAE: {best_val_mae:.2f}")

                # 记录当前 epoch 的所有最佳指标
                best_val_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'R': r,
                    'R2': r2,
                    'CCC': ccc,
                    'RMSE': rmse,
                    'MSE': mse,
                    'MAE': mae
                }

            val_rmses.append(rmse)
            val_maes.append(mae)
            val_mses.append(mse)
            val_rs.append(r)
            val_r2s.append(r2)
            val_cccs.append(ccc)

            print(f"Epoch {epoch+1}/{EPOCHS} - 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | R: {r:.4f} | R2: {r2:.4f} | CCC: {ccc:.4f} | RMSE: {rmse:.2f} | MSE: {mse:.2f} | MAE: {mae:.2f}")

    print("\n模型训练完成。")

    # --- 5. 生成并保存指标曲线图 ---
    print("正在生成性能指标图表...")

    def plot_metric(metric_values, metric_name, file_name, plot_train_val=False):
        plt.figure(figsize=(10, 6))
        if plot_train_val:
            plt.plot(range(1, EPOCHS + 1), train_losses_history, label='Training Loss', color='orange')
            plt.plot(range(1, EPOCHS + 1), val_losses_history, label='Validation Loss', color='blue')
            plt.title('Training and Validation Loss per Epoch', fontsize=16)
        else:
            plt.plot(range(1, EPOCHS + 1), metric_values, label=f'Validation {metric_name}', color='blue')
            plt.title(f'Validation {metric_name} per Epoch', fontsize=16)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, file_name))
        plt.close()

    plot_metric([], 'Loss', 'loss_curve.png', plot_train_val=True)
    plot_metric(val_rmses, 'RMSE', 'rmse_curve.png')
    plot_metric(val_maes, 'MAE', 'mae_curve.png')
    plot_metric(val_mses, 'MSE', 'mse_curve.png')
    plot_metric(val_rs, 'R', 'r_curve.png')
    plot_metric(val_r2s, 'R2', 'r2_curve.png')
    plot_metric(val_cccs, 'CCC', 'ccc_curve.png')
    print(f"性能指标图表已保存到 '{report_dir}' 目录。")

    # --- 6. 生成预测结果与真实值对比图 ---
    print("正在生成预测结果图...")

    start_idx = -200
    # 重新进行一次预测以确保绘图数据与最佳模型对应
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        val_pred_tensor = model(X_val_tensor)
        val_preds_np = val_pred_tensor.cpu().numpy()
        val_pred_unscaled = target_scaler.inverse_transform(val_preds_np)
        val_y_unscaled = target_scaler.inverse_transform(y_val_tensor.cpu().numpy())
        val_pred_unscaled[val_pred_unscaled < 0] = 0

    plot_y_true = val_y_unscaled[start_idx:]
    plot_y_pred = val_pred_unscaled[start_idx:]

    plt.figure(figsize=(15, 8))
    plt.plot(plot_y_true, color='blue', alpha=0.3)
    plt.plot(plot_y_pred, color='red', alpha=0.9, linestyle='--')
    plt.title('Prediction vs. True Values', fontsize=18)
    plt.xlabel('Time Step (60 min)', fontsize=12)
    plt.ylabel('Passenger Count', fontsize=12)
    plt.legend(['True Values (All Zones)', 'Predicted Values (All Zones)'], loc='upper left')
    plt.grid(True)
    plot_path = os.path.join(report_dir, 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"预测结果图已保存到: {plot_path}")

    # --- 7. 新增逻辑: 将最佳模型性能信息保存到文本文件 ---
    if best_val_metrics:
        report_file_path = os.path.join(report_dir, 'best_model_performance.txt')
        with open(report_file_path, 'w') as f:
            f.write("最佳模型性能报告\n")
            f.write("----------------------------\n")
            f.write(f"模型名称: {model_name}\n")
            f.write(f"最佳纪元 (Epoch): {best_val_metrics['epoch']}\n")
            f.write(f"训练损失: {best_val_metrics['train_loss']:.4f}\n")
            f.write(f"验证损失: {best_val_metrics['val_loss']:.4f}\n")
            f.write(f"R: {best_val_metrics['R']:.4f}\n")
            f.write(f"R2: {best_val_metrics['R2']:.4f}\n")
            f.write(f"CCC: {best_val_metrics['CCC']:.4f}\n")
            f.write(f"RMSE: {best_val_metrics['RMSE']:.2f}\n")
            f.write(f"MSE: {best_val_metrics['MSE']:.2f}\n")
            f.write(f"MAE: {best_val_metrics['MAE']:.2f}\n")
        print(f"最佳模型性能报告已保存到 '{report_file_path}'。")