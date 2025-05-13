# NLP-BERT-Classes

基于BERT的中文文本分类模型训练工具。该工具利用预训练的中文BERT模型进行文本分类任务的微调训练。

## 项目简介

本项目提供了一个完整的中文文本分类解决方案，使用Google的BERT-base-Chinese预训练模型作为基础，通过微调训练来适应特定领域的文本分类任务。项目特点：

- 使用预训练的中文BERT模型
- 支持多分类任务
- 内置数据预处理与平衡采样功能
- 提供详细的训练过程指标与评估
- 支持自动保存最佳模型

## 安装指南

### 1. 克隆此仓库

```bash
git clone https://github.com/yourusername/nlp-bert-classes.git
cd nlp-bert-classes
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载预训练BERT模型

请从Hugging Face下载bert-base-chinese模型：

```bash
mkdir -p bert-base-chinese
wget https://huggingface.co/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin -P bert-base-chinese
wget https://huggingface.co/google-bert/bert-base-chinese/resolve/main/config.json -P bert-base-chinese
wget https://huggingface.co/google-bert/bert-base-chinese/resolve/main/vocab.txt -P bert-base-chinese
```

或者直接访问: [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

## 使用说明

### 数据格式要求

训练数据应为Excel(.xlsx)格式，包含以下必要列：
- `question`: 待分类的文本内容
- `TITLE`: 文本标题（可选，会与问题合并）
- `label`: 分类标签

示例：

| question | TITLE | label |
|----------|-------|-------|
| 这是一个问题的内容 | 问题标题 | 类别A |
| 另一个问题的内容 | 另一个标题 | 类别B |

### 运行训练

```bash
python bert_train.py --file_path ./data/data.xlsx --model_path ./bert-base-chinese --pt_name my_classifier
```

参数说明：
- `--file_path`: 训练数据文件路径，默认为`./data/data.xlsx`
- `--model_path`: BERT预训练模型路径，默认为`./bert-base-chinese`
- `--pt_name`: 输出模型名称，默认为`nlp-bert-classes`
- `--batch_size`: 批量大小，默认为1
- `--max_len`: 最大文本长度，默认为256
- `--epochs`: 训练轮数，默认为100
- `--lr`: 学习率，默认为5e-5

### 运行预测

训练完成后，可以使用`bert_predict.py`进行预测。预测支持三种模式：批量预测文件、预测单条文本、交互式预测。

#### 批量预测文件

```bash
python bert_predict.py --model_dir ./output/my_classifier_20230101_120000 --input_file ./data/test.xlsx
```

#### 预测单条文本

```bash
python bert_predict.py --model_dir ./output/my_classifier_20230101_120000 --text "这是一个需要分类的文本内容"
```

#### 交互式预测

```bash
python bert_predict.py --model_dir ./output/my_classifier_20230101_120000
```

参数说明：
- `--bert_path`: BERT预训练模型路径，默认为`./bert-base-chinese`
- `--model_dir`: 训练好的模型目录路径（必需）
- `--model_name`: 模型名称，默认为`nlp-bert-classes`
- `--weight_type`: 使用的模型权重类型，可选`best`或`last`，默认为`best`
- `--input_file`: 输入Excel文件路径（用于批量预测）
- `--text`: 要预测的单条文本
- `--max_len`: 最大文本长度，默认为256
- `--batch_size`: 批处理大小，默认为8

### 输出说明

#### 训练输出

训练完成后，模型将保存在`output/[pt_name]_[timestamp]`目录下：
- `best_[pt_name].pth`: 验证集上表现最佳的模型权重
- `last_[pt_name].pth`: 最后一轮训练的模型权重
- `[pt_name].pkl`: 标签编码器（用于预测时将数字映射回标签）

#### 预测输出

- 批量预测：结果将保存在与输入文件同目录下的`[input_filename]_prediction.xlsx`中
- 单条文本预测：结果将直接打印到控制台
- 交互式预测：结果将实时显示在控制台

## 高级配置

更多高级配置可通过修改`TrainingConfig`类来实现，包括：
- 数据过滤阈值
- 验证集比例
- 早停策略
- 采样策略
- 等等

## 引用

如果您使用了此项目，请引用以下资源：

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

## 许可证

本项目遵循 LICENSE 文件中指定的许可条款。

## 感谢🙏
- [google-bert](https://huggingface.co/google-bert): bert预训练模型。
- [pytorch](https://pytorch.org/): 深度学习框架。
