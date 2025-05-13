import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import time
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.utils import resample
from dataclasses import dataclass
from typing import Optional
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import os
import argparse
from datetime import datetime


@dataclass
class TrainingConfig:
    """训练配置类"""
    file_path: str  # 数据集路径
    model_path: str  # bert模型路径
    pt_name: str  # 保存模型名称
    weight_path: Optional[str] = ''  # 继续训练的权重文件地址
    max_len: int = 256  # 最大文本长度
    batch_size: int = 1  # 每批训练数据量
    filter_count: int = 40  # 过滤小于此数的类别数据
    test_size: float = 0.1  # 训练集和验证集比例
    epochs: int = 100  # 最大训练轮数
    patience: int = 4  # 验证loss不再下降时,训练终止
    lr: float = 5e-5  # 学习率
    sampling_ratio: float = 0.5  # 采样比例
    sampling_filter: int = 35000  # 大于此值才进行下采样


"""模型训练主函数"""
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextClassifier(nn.Module):
    def __init__(self, n_classes, model_path):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(last_hidden_state)
        return self.out(output[:, 0, :])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


class MultiClassReport():
    """
    Accuracy, F1 Score, Precision and Recall for multi-class classification task.
    average：micro、macro
    """

    def __init__(self, name='MultiClassReport', average='macro', n_classes=None):
        super(MultiClassReport, self).__init__()
        self.average = average
        self._name = name
        # 创建评估指标对象
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=n_classes).to(device)
        # f1招回和精准度的调和平均值
        self.f1_metric = F1Score(task='multiclass', num_classes=n_classes, average='macro').to(device)
        # 精确度
        self.precision_metric = Precision(task='multiclass', num_classes=n_classes, average='macro').to(device)
        # 召回率
        self.recall_metric = Recall(task='multiclass', num_classes=n_classes, average='macro').to(device)
        # 混淆矩阵 (对角线之和就是预测对的数目)
        self.confusion_metric = ConfusionMatrix(task='multiclass', num_classes=n_classes).to(device)

    def reset(self):
        """
        Resets all the metric state.
        """
        self.accuracy_metric.reset()
        self.f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.confusion_metric.reset()

    def update(self, probs, labels):
        self.accuracy_metric.update(probs, labels)
        self.f1_metric.update(probs, labels)
        self.precision_metric.update(probs, labels)
        self.recall_metric.update(probs, labels)
        self.confusion_metric.update(probs, labels)

    def accumulate(self):
        accuracy = self.accuracy_metric.compute()
        f1 = self.f1_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        confusion = self.confusion_metric.compute()
        return accuracy, f1, precision, recall, confusion

    def name(self):
        """
        Returns metric name
        """
        return self._name


def loss_weight(class_num, pixel_count):
    """计算分类权重"""
    W = 1 / np.log(pixel_count)
    W = class_num * W / np.sum(W)
    return W


# 排除效果不好labels
remove_labels = ['remove_label']


def prepare_dataset(config: TrainingConfig, output_dir: str):
    """准备数据集"""
    # 读取数据
    df = pd.read_excel(config.file_path, engine='openpyxl', sheet_name=None)
    df = pd.concat(df, ignore_index=True)

    # 数据清洗
    df.dropna(subset=['question'], inplace=True)
    df.drop_duplicates(subset=['question'], keep='first', inplace=True)
    df = df[df['question'].str.len() > 30]

    df = df[~df['label'].isin(df['label'].value_counts()[df['label'].value_counts() < config.filter_count].index)]
    df = df[~df['label'].isin(remove_labels)]
    # 获取标签统计
    label_counts = df['label'].value_counts()
    # 打印数据分布情况
    print("Label counts:", label_counts, flush=True)
    # 合并标题和问题
    df["question"] = df["question"].astype(str) + "[SEP]" + df["TITLE"].astype(str)
    # df["question"] = df["question"].astype(str)

    # 标签编码
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    joblib.dump(encoder, os.path.join(output_dir, f'{config.pt_name}.pkl'))

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        df['question'],
        df['label'],
        test_size=config.test_size,
        random_state=42,
        stratify=df['label']
    )

    return (X_train.reset_index(drop=True), y_train.reset_index(drop=True),
            X_val.reset_index(drop=True), y_val.reset_index(drop=True),
            label_counts)


def data_samples(X_train, y_train, config: TrainingConfig):
    """数据采样"""
    class_counts = np.bincount(y_train)

    # 计算采样权重
    weights = 1 / np.log1p(class_counts)  # 使用log1p避免log(1)=0的情况
    n_samples = np.floor(class_counts * weights).astype(int)

    X_resampled = []
    y_resampled = []

    for class_label in range(len(class_counts)):
        class_indices = np.where(y_train == class_label)[0]
        X_class_samples = X_train[class_indices]
        y_class_samples = y_train[class_indices]

        n_samples_class = n_samples[class_label]
        # 保护小样本
        if len(class_indices) < config.sampling_filter:
            n_samples_class = len(class_indices)
        else:
            n_samples_class = config.sampling_filter

        X_resampled_class, y_resampled_class = resample(
            X_class_samples,
            y_class_samples,
            replace=False,
            n_samples=n_samples_class,
            random_state=42
        )

        X_resampled.append(X_resampled_class)
        y_resampled.append(y_resampled_class)

    return pd.Series(np.concatenate(X_resampled)), pd.Series(np.concatenate(y_resampled))


def train_epoch(model, train_data_loader, optimizer, loss_fn, device, pbar):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_postfix(loss=f'{total_loss / int(pbar.n + 1):.4f}')

    return total_loss / len(train_data_loader)


def validate_epoch(model, val_data_loader, loss_fn, device, pbar, micro_report):
    """验证一个epoch"""
    model.eval()
    total_val_accuracy = 0
    total_val_loss = 0
    micro_report.reset()

    with torch.no_grad():
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()

            _, predicted_labels = torch.max(outputs, dim=1)
            total_val_accuracy += torch.sum(predicted_labels == labels).item() / len(labels)
            pbar.update(1)
            pbar.set_postfix(
                loss=f'{total_val_loss / int(pbar.n + 1) :.2f}, accuracy=:{total_val_accuracy / int(pbar.n):.2f}')
            micro_report.update(predicted_labels, labels)

    # 计算真实标签和预测标签之间的混淆矩阵及指标
    accuracy, f1, precision, recall, confusion = micro_report.accumulate()
    micro_report.reset()
    return total_val_loss / len(val_data_loader), accuracy, f1, precision, recall, confusion


def train_model(config: TrainingConfig):
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"{config.pt_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"模型将保存到: {output_dir}")
    
    # 准备数据
    X_train, y_train, X_val, y_val, label_counts = prepare_dataset(config, output_dir)
    # 验证集数据下采样
    X_train, y_train = data_samples(X_train, y_train, config)

    encoder = joblib.load(os.path.join(output_dir, f'{config.pt_name}.pkl'))
    print(f"-------------------------\nAfter Label counts")
    for item in y_train.value_counts().items():
        print(f"{encoder.inverse_transform([item[0]])[0]}\t\t {item[1]}", flush=True)
    micro_report = MultiClassReport(n_classes=len(encoder.classes_))

    # 创建数据加载器
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    train_dataset = TextDataset(X_train.to_numpy(), y_train, tokenizer, config.max_len)
    val_dataset = TextDataset(X_val.to_numpy(), y_val, tokenizer, config.max_len)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
                                   num_workers=0)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=0)

    # 初始化模型
    model = TextClassifier(len(np.unique(y_train)), config.model_path)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 设置损失函数和优化器
    pixel_count = y_train.value_counts().values * 1000
    class_weights = torch.tensor(loss_weight(len(y_train.value_counts()), pixel_count), dtype=torch.float32)

    # loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device)
    loss_fn = FocalLoss().to(device)

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.01},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    # ]
    optimizer = AdamW(model.parameters(), lr=config.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1)

    # 训练循环
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(config.epochs):
        # 训练
        pbar = tqdm(total=len(train_data_loader), desc=f"Training: {epoch + 1}", position=0, leave=True)
        train_loss = train_epoch(model, train_data_loader, optimizer, loss_fn, device, pbar)
        pbar.close()
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.2f}')

        # 验证
        pbar2 = tqdm(total=len(val_data_loader), desc=f"Validation: {epoch + 1}", position=0, leave=True)
        val_loss, accuracy, f1, precision, recall, confusion = validate_epoch(model, val_data_loader, loss_fn,
                                                                              device, pbar2, micro_report)
        pbar2.close()
        # 打印模型评估指标
        print(
            f'Validation Loss: {val_loss:.2f}\naccuracy: {accuracy:.2f}\nF1: {f1:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}',
            flush=True)
        for i, cf in enumerate(confusion):
            max_val, max_index = torch.topk(cf, 1, dim=0)
            label_cons = cf.sum().item()
            print(
                f'label:{encoder.inverse_transform([i])[0]} raio:{cf[i].item() / label_cons * 100:.2f}%  max_label:{encoder.inverse_transform([max_index.item()])[0]} max_raio:{max_val.item() / label_cons * 100:.2f}%',
                flush=True)
        # 学习率调整
        # scheduler.step(val_loss)
        # print(f'Learning Rate: {scheduler.get_last_lr()[0]:.8f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_{config.pt_name}.pth'))
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= config.patience:
                print("Early stopping.")
                break

        torch.save(model.state_dict(), os.path.join(output_dir, f'last_{config.pt_name}.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='BERT中文文本分类训练')
    parser.add_argument('--file_path', type=str, default='./data/data.xlsx', help='数据集路径')
    parser.add_argument('--model_path', type=str, default='./bert-base-chinese', help='BERT模型路径')
    parser.add_argument('--pt_name', type=str, default='nlp-bert-classes', help='保存模型名称')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--max_len', type=int, default=256, help='最大文本长度')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainingConfig(
        file_path=args.file_path,
        model_path=args.model_path,
        pt_name=args.pt_name,
        batch_size=args.batch_size,
        max_len=args.max_len,
        epochs=args.epochs,
        lr=args.lr
    )
    train_model(config)
