import os
import warnings
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from copy import deepcopy

# ========== 全局配置：仅调参提准确率，核心算法不变 ==========
CONFIG = {
    "excel_path": "Ra.xlsx",
    "n_signals": 1,  # 保持单路信号
    "n_samples_per_label": 50,  # 从 20 改成 50
    "target_win_size": 80,
    "filter_size": 5,
    "peak_valley_distance": 30,
    "peak_valley_prominence": 0.3,
    "similarity_threshold": 0.6,
    "adjust_step": 5,
    "candidate_range": 20,
    "n_splits": 5,
    "random_state": 42,
    "epochs": 80,
    "batch_size": 64,
    "test_batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "patience": 15,
    "factor": 0.5,
    "hidden_size": 256,
    "lstm_layers": 1,
    "vis_samples": list(range(10)),
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "augment_noise": 0.01
}

# ========== 初始化 ==========
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_metrics = {}
scaler = GradScaler()

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['figure.titlesize'] = 14


# ========== 工具函数 ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭，保证复现


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_mean_and_std(list_of_folds):
    arr = np.array(list_of_folds)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        x, _ = batch
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds)


# ========== 峰峰值提取算法【完全未修改】 ==========
def find_valleys(signal, config):
    neg_signal = -signal
    valleys, _ = find_peaks(
        neg_signal,
        distance=config["peak_valley_distance"],
        prominence=config["peak_valley_prominence"]
    )
    return valleys


def find_peaks_signal(signal, config):
    peaks, _ = find_peaks(
        signal,
        distance=config["peak_valley_distance"],
        prominence=config["peak_valley_prominence"]
    )
    return peaks


def get_peak_to_peak_waveforms(signal, peaks, valleys, config):
    waveforms = []
    peak_to_peak_values = []
    valley_idx = 0
    peak_idx = 0

    while valley_idx < len(valleys) - 1 and peak_idx < len(peaks):
        while peak_idx < len(peaks) and peaks[peak_idx] < valleys[valley_idx]:
            peak_idx += 1
        if peak_idx >= len(peaks):
            break

        while valley_idx < len(valleys) - 1 and valleys[valley_idx + 1] < peaks[peak_idx]:
            valley_idx += 1
        if valley_idx + 1 >= len(valleys):
            break

        start_valley = valleys[valley_idx]
        mid_peak = peaks[peak_idx]
        end_valley = valleys[valley_idx + 1]
        waveform = signal[start_valley:end_valley + 1]

        valley_min = min(signal[start_valley], signal[end_valley])
        peak_max = signal[mid_peak]
        peak_to_peak = peak_max - valley_min

        if len(waveform) >= config["target_win_size"] // 2:
            waveforms.append(waveform)
            peak_to_peak_values.append(peak_to_peak)

        valley_idx += 1
        peak_idx += 1

    return np.array(waveforms, dtype=object), np.array(peak_to_peak_values)


def generate_peak_to_peak_windows(waveforms, peak_to_peak_values, config):
    target_n = config["n_samples_per_label"]
    target_len = config["target_win_size"]
    windows = []

    if len(waveforms) >= target_n:
        ppt_mean = np.mean(peak_to_peak_values)
        ppt_var = np.square(peak_to_peak_values - ppt_mean)
        selected_idx = np.argsort(ppt_var)[:target_n]
        selected_waveforms = [waveforms[i] for i in selected_idx]
    else:
        ppt_mean = np.mean(peak_to_peak_values) if len(peak_to_peak_values) > 0 else 0
        closest_idx = np.argmin(np.abs(peak_to_peak_values - ppt_mean)) if len(peak_to_peak_values) > 0 else 0
        supplement_num = target_n - len(waveforms)
        selected_waveforms = list(waveforms) + (
            [waveforms[closest_idx]] * supplement_num if len(waveforms) > 0 else [np.zeros(target_len)])

    for wf in selected_waveforms:
        interp_wf = np.interp(
            np.linspace(0, len(wf) - 1, target_len),
            np.arange(len(wf)),
            wf
        )
        windows.append(interp_wf.astype(np.float32))

    return np.array(windows)


# ========== 可视化函数【完全未修改】 ==========
def plot_peak_to_peak_windows(signal_data, peaks, valleys, waveforms, windows, label, config, vis_samples):
    n_vis = len(vis_samples)
    fig, axes = plt.subplots(1 + n_vis, 1, figsize=(12, 2 * (1 + n_vis)))
    fig.suptitle(f"Peak-to-Peak Signal Processing - Material: {label}", fontweight='bold')

    ax = axes[0]
    ax.plot(signal_data, color='steelblue', linewidth=0.8, label='Filtered Signal')
    ax.scatter(valleys, signal_data[valleys], color='gray', s=15, zorder=3, label='Valley')
    ax.scatter(peaks, signal_data[peaks], color='blue', marker='^', s=20, zorder=3, label='Peak')
    for wf in waveforms[:5]:
        start_idx = np.where(signal_data == wf[0])[0][0] if len(wf) > 0 else 0
        ax.plot(range(start_idx, start_idx + len(wf)), wf, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Voltage (mV)", fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.legend(fontsize=8, loc='upper right')

    for row_idx, sample_idx in enumerate(vis_samples):
        if sample_idx >= len(windows):
            continue
        ax = axes[row_idx + 1]
        window = windows[sample_idx]
        ax.plot(window, color='crimson', linewidth=1.0, label=f'P2 Sample {sample_idx + 1}')
        ax.set_xlabel("Window Time Points", fontsize=10)
        ax.set_ylabel("Voltage (mV)", fontsize=10)
        ax.grid(alpha=0.2, linewidth=0.3)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.6)
    plt.show()


# ========== 数据预处理【核心峰峰值算法完全不变，只修复数组报错】 ==========
def load_and_preprocess_data(config):
    data = pd.read_excel(config["excel_path"], header=None)
    col_names = ['t'] + ['value1'] + ['label']
    data.columns = col_names[:len(data.columns)]

    data['value1'] = pd.to_numeric(data['value1'], errors='coerce')
    data = data.dropna()

    labels = data['label'].unique()
    if len(labels) == 0:
        raise RuntimeError('数据为空')
    print(f"检测到 {len(labels)} 种材料")

    raw_segments = []
    labels_order = []

    for label in tqdm(labels, desc="处理每种材料"):
        sub_data = data[data['label'] == label].reset_index(drop=True)
        if sub_data.empty:
            warnings.warn(f'{label} 无数据')
            continue

        sig = sub_data['value1'].values
        sig_filt = uniform_filter1d(sig, size=config["filter_size"])
        sig_scaled = StandardScaler().fit_transform(sig_filt.reshape(-1, 1)).flatten()

        peaks = find_peaks_signal(sig_scaled, config)
        valleys = find_valleys(sig_scaled, config)

        waveforms, ppt_values = get_peak_to_peak_waveforms(sig_scaled, peaks, valleys, config)
        windows = generate_peak_to_peak_windows(waveforms, ppt_values, config)
        n_samples = min(len(windows), config["n_samples_per_label"])

        avg_ppt = np.mean(ppt_values) if len(ppt_values) > 0 else 0.0
        print(f"{label}：{n_samples} 样本，平均峰峰值{avg_ppt:.4f}")

        for i in range(n_samples):
            sample = np.expand_dims(windows[i], axis=0)
            raw_segments.append(sample)
            labels_order.append(label)

        vis_samples = [idx for idx in config["vis_samples"] if idx < n_samples]
        # plot_peak_to_peak_windows(sig_scaled, peaks, valleys, waveforms, windows, label, config, vis_samples)

    # ====================== 修复报错的核心位置 ======================
    # 强制统一数组形状，避免 inhomogeneous shape 错误
    raw_segments = np.asarray(raw_segments, dtype=np.float32)
    labels_order = np.array(labels_order)
    # ===============================================================

    print(f'总样本：{len(raw_segments)}，维度：{raw_segments.shape}')
    return raw_segments, labels_order, data


# ========== 数据集：加入微小噪声增强（不破坏峰峰值） ==========
class CNNDataset(Dataset):
    def __init__(self, x, y, noise=0.01, train=True):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.noise = noise if train else 0.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        return x, self.y[idx]


# ========== CNN-LSTM模型结构【完全未修改】 ==========
class CNN_LSTM_Res(nn.Module):
    def __init__(self, n_class, hidden=256, layers=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Linear(hidden * 2, n_class)

    def forward(self, x):
        c = self.pool(self.conv1(x))
        c = self.pool(self.conv2(c))
        c = c.transpose(1, 2)
        out_lstm, _ = self.lstm(c)
        out = out_lstm[:, -1, :]
        return self.classifier(out)


# ========== 训练函数 ==========
def train_evaluate_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    best_acc = 0
    best_state = None
    fold_train_losses = []
    fold_val_losses = []
    fold_val_accs = []
    early_stop_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast():
                x, yb = batch
                x, yb = x.to(device), yb.to(device)
                outputs = model(x)
                loss = criterion(outputs, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item() * yb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        fold_train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                x, yb = batch
                x, yb = x.to(device), yb.to(device)
                outputs = model(x)
                loss = F.cross_entropy(outputs, yb)
                total_val_loss += loss.item() * yb.size(0)
                pred = outputs.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        avg_val_loss = total_val_loss / len(test_loader.dataset)
        acc_val = correct / total
        fold_val_losses.append(avg_val_loss)
        fold_val_accs.append(acc_val)

        scheduler.step()

        if acc_val > best_acc:
            best_acc = acc_val
            best_state = deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config["patience"]:
            print(f"早停：Epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | TrainLoss {avg_train_loss:.4f} | ValAcc {acc_val:.4f}")

    return best_acc, best_state, fold_train_losses, fold_val_losses, fold_val_accs


# ========== 交叉验证 ==========
def cross_validate(model_fn, dataset, config, class_names):
    all_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    kfold = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    acc_list = []
    y_true_all = []
    y_pred_all = []
    train_losses = []
    val_losses = []
    val_accs = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(dataset)), all_labels), 1):
        print(f'\n>>> 第 {fold}/{config["n_splits"]} 折')

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        g = torch.Generator()
        g.manual_seed(config["random_state"])
        num_workers = 0 if os.name == 'nt' else 4

        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"], shuffle=True,
            pin_memory=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
        )
        test_loader = DataLoader(
            test_subset, batch_size=config["test_batch_size"],
            pin_memory=True, num_workers=num_workers
        )

        model = model_fn(n_class=len(class_names), hidden=config["hidden_size"]).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

        best_acc, best_state, fold_train_loss, fold_val_loss, fold_val_acc = train_evaluate_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, config
        )

        acc_list.append(best_acc)
        train_losses.append(fold_train_loss)
        val_losses.append(fold_val_loss)
        val_accs.append(fold_val_acc)

        model.load_state_dict(best_state)
        y_pred_fold = predict(model, test_loader, device)
        y_true_fold = all_labels[test_idx]
        y_true_all.extend(y_true_fold)
        y_pred_all.extend(y_pred_fold)

    print(f'\n5折准确率: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')

    try:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true_all,
            y_pred_all,
            display_labels=class_names,
            cmap='Blues',
            text_kw={'fontsize':14}  # 矩阵内数字字体14
        )
        disp.ax_.set_title('CNN-LSTM', fontsize=16)
        # 坐标轴标签字体
        disp.ax_.xaxis.label.set_fontsize(16)
        disp.ax_.yaxis.label.set_fontsize(16)
        # 刻度标签字体
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()
    except:
        print("混淆矩阵绘制失败，跳过")

    report = classification_report(y_true_all, y_pred_all, output_dict=True)
    model_metrics['CNN-LSTM'] = {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }
    return train_losses, val_losses, val_accs


# ========== 主函数 ==========
# ========== 主函数 ==========
def main():
    set_seed(CONFIG["random_state"])
    raw_segments, labels_order, _ = load_and_preprocess_data(CONFIG)

    # ✅ 强制使用固定标签顺序编码
    fixed_label_order = ["Ra0.6", "Ra2.3", "Ra6.3", "Ra12.3"]
    le = LabelEncoder()
    le.classes_ = np.array(fixed_label_order)
    y_encoded = le.transform(labels_order)
    class_names = le.classes_

    shuffle_idx = np.arange(len(raw_segments))
    np.random.shuffle(shuffle_idx)
    X_raw = raw_segments[shuffle_idx]
    y_raw = y_encoded[shuffle_idx]

    dataset = CNNDataset(X_raw, y_raw, noise=CONFIG["augment_noise"], train=True)

    print('\n=== 模型训练 ===')
    train_losses, val_losses, val_accs = cross_validate(CNN_LSTM_Res, dataset, CONFIG, class_names)

    print("\n--- 最终指标 ---")
    m = model_metrics['CNN-LSTM']
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1:       {m['f1-score']:.4f}")

    # ====================== 修复训练曲线绘制（必出图） ======================
    try:
        import matplotlib.pyplot as plt
        max_epoch = max(len(t) for t in train_losses)

        padded_train = []
        padded_val = []
        padded_acc = []

        for tl, vl, acc in zip(train_losses, val_losses, val_accs):
            pad = max_epoch - len(tl)
            padded_train.append(tl + [np.nan] * pad)
            padded_val.append(vl + [np.nan] * pad)
            padded_acc.append(acc + [np.nan] * pad)

        train_mean = np.nanmean(padded_train, axis=0)
        train_std = np.nanstd(padded_train, axis=0)
        val_mean = np.nanmean(padded_val, axis=0)
        val_std = np.nanstd(padded_val, axis=0)
        acc_mean = np.nanmean(padded_acc, axis=0)
        acc_std = np.nanstd(padded_acc, axis=0)

        epochs = list(range(1, max_epoch + 1))
        plt.figure(figsize=(10, 5))

        plt.plot(epochs, train_mean, 'b-', linewidth=1.5, label='Train Loss')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')

        plt.plot(epochs, val_mean, 'g-', linewidth=1.5, label='Val Loss')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')

        plt.twinx()
        plt.plot(epochs, acc_mean, 'r-', linewidth=1.5, label='Val Acc')
        plt.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color='red')
        plt.ylim(0, 1.05)

        plt.title('Training Curve (5-Fold Average)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy', color='red')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("训练曲线绘制失败，原因：", str(e))

if __name__ == "__main__":
    main()