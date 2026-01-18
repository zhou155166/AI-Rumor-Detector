import torch
import torch.nn as nn
import joblib
import re

# BiGRU 类定义
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        out = self.fc(h)
        return out.squeeze(1)

class RumourDetectClass:
    def __init__(self):
        # 指定模型文件路径
        model_path = r"F:\大学\大二下\人工智能导论（A类）\大作业\我的代码\bigru_model.pth"
        vocab_path = r"F:\大学\大二下\人工智能导论（A类）\大作业\我的代码\vocab.pkl"

        # 加载词表
        self.vocab = joblib.load(vocab_path)

        # 加载模型
        self.model = BiGRU(len(self.vocab), 100, 128)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print("模型加载成功")

    def preprocess(self, text):
        # 与训练时一致的小写和去标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def classify(self, text: str) -> tuple:
        """
        对输入的文本进行谣言检测，并返回预测的类别和概率
        Args:
            text: 输入的文本字符串
        Returns:
            tuple: (预测的类别（0表示非谣言，1表示谣言），预测的概率)
        """
        text = self.preprocess(text)
        tokens = [self.vocab.get(t, self.vocab['<UNK>']) for t in re.findall(r'\w+', text.lower())]
        if len(tokens) < 64:
            tokens += [self.vocab['<PAD>']] * (64 - len(tokens))
        else:
            tokens = tokens[:64]
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred_prob = torch.sigmoid(logits).item()
            pred = int(pred_prob > 0.5)  # 修正布尔值转换为整数
        return pred, pred_prob

# 测试代码
if __name__ == '__main__':
    detector = RumourDetectClass()
    test_texts = [
        "This is a normal tweet.",
        "Breaking news: a major event has happened!",
        "This is a false alarm, nothing to worry about.",
        "Exclusive: a celebrity has been secretly dating another celebrity!",
        "Urgent: a new virus has been discovered and is spreading rapidly!",
        "Fake news: a politician has been involved in a major scandal!"
    ]

    for text in test_texts:
        result, prob = detector.classify(text)
        print(f"Text: '{text}' -> Prediction: {result}, Probability: {prob:.4f}")