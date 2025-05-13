# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import argparse
import joblib
import pandas as pd
import os
from tqdm import tqdm


class TextClassifier(nn.Module):
    def __init__(self, n_classes, model_path):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False  # 预测时不需要梯度
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


def predict_text(text, model, tokenizer, max_len, device):
    """预测单个文本"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
    return preds.item()


def predict_batch(texts, model, tokenizer, max_len, device, batch_size=8):
    """批量预测文本"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
        results.extend(preds.cpu().numpy().tolist())
    
    return results


def main(args):
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载标签编码器
    encoder_path = os.path.join(args.model_dir, f"{args.model_name}.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"找不到标签编码器: {encoder_path}")
    
    encoder = joblib.load(encoder_path)
    n_classes = len(encoder.classes_)
    print(f"分类类别数: {n_classes}")
    print(f"分类标签: {encoder.classes_}")
    
    # 加载模型
    model_path = os.path.join(args.model_dir, f"{args.weight_type}_{args.model_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")
    
    # 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = TextClassifier(n_classes, args.bert_path)
    
    # 加载模型权重
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"已加载模型: {model_path}")
    
    # 预测
    if args.input_file:
        # 从文件预测
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"找不到输入文件: {args.input_file}")
        
        # 读取文件
        df = pd.read_excel(args.input_file, engine='openpyxl')
        
        if 'question' not in df.columns:
            raise ValueError("输入文件必须包含'question'列")
        
        if 'TITLE' in df.columns:
            df['text'] = df['question'].astype(str) + "[SEP]" + df['TITLE'].astype(str)
        else:
            df['text'] = df['question']
        
        texts = df['text'].tolist()
        print(f"正在预测 {len(texts)} 条文本...")
        
        # 批量预测
        results = []
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[i:i+args.batch_size]
            batch_results = predict_batch(batch_texts, model, tokenizer, args.max_len, device, args.batch_size)
            results.extend(batch_results)
        
        # 将预测结果转换为原始标签
        label_results = encoder.inverse_transform(results)
        
        # 保存结果
        df['predicted_label'] = label_results
        output_path = f"{os.path.splitext(args.input_file)[0]}_prediction.xlsx"
        df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")
        
    elif args.text:
        # 预测单个文本
        pred = predict_text(args.text, model, tokenizer, args.max_len, device)
        label = encoder.inverse_transform([pred])[0]
        print(f"预测结果: {label}")
    else:
        # 交互式预测
        print("进入交互式预测模式，输入 'exit' 退出")
        while True:
            text = input("请输入要预测的文本: ")
            if text.lower() == 'exit':
                break
            
            pred = predict_text(text, model, tokenizer, args.max_len, device)
            label = encoder.inverse_transform([pred])[0]
            print(f"预测结果: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT中文文本分类预测")
    parser.add_argument("--bert_path", type=str, default="./bert-base-chinese", help="BERT预训练模型路径")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--model_name", type=str, default="nlp-bert-classes", help="模型名称")
    parser.add_argument("--weight_type", type=str, default="best", choices=["best", "last"], help="使用最佳模型还是最后一个模型")
    parser.add_argument("--input_file", type=str, help="输入Excel文件路径")
    parser.add_argument("--text", type=str, help="预测单个文本")
    parser.add_argument("--max_len", type=int, default=256, help="最大文本长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    
    args = parser.parse_args()
    main(args)