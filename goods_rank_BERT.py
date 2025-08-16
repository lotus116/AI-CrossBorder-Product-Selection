import pandas as pd
import torch
import os
import glob
import re
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split


# -------------------------- 数据预处理 --------------------------
def preprocess_data(input_path='./ratings.csv', output_path='./ratings_dealed.csv'):
    """预处理原始数据，生成三分类标签并保存"""
    print("开始数据预处理...")
    
    # 读取并筛选数据
    ratings_origin = pd.read_csv(input_path)
    ratings_dealed = ratings_origin.loc[:, ['comment', 'rating']].dropna()  # 直接删除空值
    print(f'已处理comment空值个数: {ratings_origin["comment"].isnull().sum()}')

    # 过滤异常值并转换为整数
    ratings_dealed = ratings_dealed[ratings_dealed['rating'] != 0]
    ratings_dealed['rating'] = ratings_dealed['rating'].astype(int)

    # 转换为三分类标签（0=差评，1=中评，2=好评）
    ratings_dealed['rating'] = ratings_dealed['rating'].apply(
        lambda x: 0 if x in [1, 2] else 1 if x == 3 else 2
    )

    # 查看标签分布
    print("\n三分类标签分布：")
    print(ratings_dealed['rating'].value_counts())

    # 保存处理后的数据
    ratings_dealed.to_csv(output_path, index=False)
    print(f'处理后的数据已保存至: {output_path}\n')
    return output_path


# -------------------------- 数据集类 --------------------------
class BertDataset(Dataset):
    """BERT专用数据集，转换文本为模型输入格式"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.flatten() for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -------------------------- 模型训练与保存 --------------------------
def train_and_save(
    data_path='./ratings_dealed.csv',
    model_base_path='./bert-base-chinese',
    save_base_path='./fine_tuned_bert_model',
    train_sample_size=10000,
    val_sample_size=2000,
    batch_size=8,
    num_epochs=3,
    max_len=128
):
    """训练模型并保存（含时间戳）"""
    # 1. 加载处理后的数据
    ratings_dealed = pd.read_csv(data_path)
    print(f"数据形状：{ratings_dealed.shape}")
    print(ratings_dealed.head(2))

    # 2. 拆分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        ratings_dealed['comment'].tolist(),
        ratings_dealed['rating'].tolist(),
        test_size=0.2,
        random_state=42
    )
    # 截取样本（控制训练规模）
    train_texts = train_texts[:train_sample_size]
    train_labels = train_labels[:train_sample_size]
    val_texts = val_texts[:val_sample_size]
    val_labels = val_labels[:val_sample_size]
    print(f"训练集样本数：{len(train_texts)}，验证集样本数：{len(val_texts)}")

    # 3. 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_base_path)
    model = BertForSequenceClassification.from_pretrained(
        model_base_path,
        num_labels=3  # 三分类
    )

    # 4. 创建数据加载器
    train_dataset = BertDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = BertDataset(val_texts, val_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 5. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备：{device}\n")

    # 6. 训练模型
    optimizer = AdamW(model.parameters(), lr=2e-5)
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(** batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                total_val_acc += (preds == batch['labels']).float().mean().item()
        avg_val_acc = total_val_acc / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失：{avg_train_loss:.4f} | 验证准确率：{avg_val_acc:.4f}")

    # 7. 保存模型（带时间戳）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_base_path}/goods_rank_model_{current_time}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存至：{save_path}")
    return save_path


# -------------------------- 辅助函数：获取最新模型路径 --------------------------
def get_latest_model_path(base_dir='./fine_tuned_bert_model'):
    """自动查找最新训练的模型路径（按时间戳排序）"""
    # 查找所有符合命名规则的模型文件夹（含时间戳）
    pattern = os.path.join(base_dir, 'goods_rank_model_*')
    model_dirs = glob.glob(pattern)
    
    if not model_dirs:
        return None  # 无模型文件夹
    
    # 提取时间戳并排序（最新的在最后）
    def extract_timestamp(path):
        match = re.search(r'goods_rank_model_(\d{8}_\d{6})', path)
        return match.group(1) if match else ''
    
    # 按时间戳降序排序，取第一个即为最新
    model_dirs.sort(key=extract_timestamp, reverse=True)
    return model_dirs[0]


# -------------------------- 模型加载与预测 --------------------------
def load_and_predict(model_path):
    """加载已训练的模型并进行预测"""
    # 1. 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型加载完成（设备：{device}）")

    # 2. 预测函数
    def predict(comment):
        encoding = tokenizer(
            comment,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(** {k: v.to(device) for k, v in encoding.items()})
            pred_label = torch.argmax(outputs.logits, dim=1).item()
        return ["差评（1-2分）", "中评（3分）", "好评（4-5分）"][pred_label]

    # 3. 示例预测
    test_comments = [
        "这个商品太棒了，性价比很高，推荐购买！",
        "一般般，没有特别惊喜的地方",
        "很差劲，质量不行，不建议买",
        "不吹不黑，只能品质说对得起这个价格吧",
        "肯德基疯狂星期四，感觉性价比一般"
    ]
    print("\n示例预测：")
    for comment in test_comments:
        print(f"评论：{comment} → 预测：{predict(comment)}")

    # 4. 交互式预测
    while True:
        user_input = input("\n请输入评论（输入q退出）：")
        if user_input.lower() == 'q':
            break
        print(f"预测结果：{predict(user_input)}")


# -------------------------- 主函数（交互入口） --------------------------
def main():
    print("===== BERT评论情感分析工具 =====")
    print("1. 训练模型并保存")
    print("2. 加载模型并预测")
    choice = input("请选择功能（1/2）：")

    if choice == '1':
        # 训练流程：可直接使用默认参数，或根据需求修改
        #===============================================
        # preprocess_data()  # 【预处理数据（若已处理可注释）】
        # ==============================================
        start_time = datetime.now()
        print('开始训练:',start_time.strftime("%Y%m%d_%H%M%S"))

        train_and_save(
            train_sample_size=5000,  # 可调整训练样本数
            val_sample_size=1000,
            num_epochs=3  # 可调整训练轮数
        )

        end_time = datetime.now()
        print('训练结束:', end_time.strftime("%Y%m%d_%H%M%S"))

        # 计算耗时
        time_diff = end_time - start_time
        minutes, seconds = divmod(time_diff.seconds, 60)
        print(f'总共耗时: {minutes}分{seconds}秒')

    elif choice == '2':
        # 预测流程：支持默认最新模型或用户输入路径
        model_path_input = input(
            "请输入模型保存路径（直接回车则使用最新模型）：\n"
            "(例如./fine_tuned_bert_model/goods_rank_model_20240520_123045）："
        ).strip()

        # 确定模型路径
        if not model_path_input:  # 用户未输入，自动找最新模型
            print("正在查找最新训练的模型...")
            model_path = get_latest_model_path()
            if not model_path:
                print("未找到任何训练好的模型，请先执行训练（选择1）")
                return
            print(f"已自动选择最新模型：{model_path}")
        else:  # 用户输入了路径
            model_path = model_path_input
            if not os.path.exists(model_path):
                print("路径不存在，请检查后重试")
                return

        # 加载模型并预测
        load_and_predict(model_path)

    else:
        print("无效选择")


if __name__ == "__main__":
    main()