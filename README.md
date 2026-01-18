仅放了我的部分，图片没法防止readme
第一步：软件准备
Windows下win+R输入cmd回车进入命令提示符
在官网下找小于2.0.0选1.8.2：conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts后发现pytorch版本竟然不对，是因为python版本不是低于3.8
卸载：conda remove pytorch torchvision torchaudio和conda remove cudatoolkit再conda clean --all
重试：conda create -n new_env python=3.8和conda activate new_env并conda init后
发现python版本是降低了
 但是下载pytorch会超时。考虑使用国内的镜像源，如清华大学的镜像源：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
重新conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
成功！
第二步：代码试运行
train_model.py:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 指定数据文件路径
data_path = r"F:\大学\大二下\人工智能导论（A类）\大作业\谣言数据集\\"  # 注意路径末尾的反斜杠

# 读取数据
train_df = pd.read_csv(data_path + 'train.csv')
val_df = pd.read_csv(data_path + 'val.csv')

# 特征和标签
X_train = train_df['text']
y_train = train_df['label']
X_val = val_df['text']
y_val = val_df['label']

# 文本预处理
X_train = X_train.str.lower().str.replace('[^\w\s]', '', regex=True)
X_val = X_val.str.lower().str.replace('[^\w\s]', '', regex=True)

# 文本向量化（TF-IDF）
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 验证集评估
val_pred = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_pred)
print(f'Val Acc: {val_acc:.4f}')
print(classification_report(y_val, val_pred))

# 保存模型和向量器
model_save_path = r"F:\大学\大二下\人工智能导论（A类）\大作业\我的代码\\"  # 注意路径末尾的反斜杠
joblib.dump({'model': model, 'vectorizer': vectorizer}, model_save_path + 'lr_model.pkl')
print('模型已保存为lr_model.pkl')

classify.py:
import joblib
import re

class RumourDetectClass:
    def __init__(self):
        # 指定模型文件路径
        model_path = r"F:\大学\大二下\人工智能导论（A类）\大作业\我的代码\lr_model.pkl"
        
        # 加载模型和vectorizer
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        print("模型加载成功")

    def preprocess(self, text):
        # 与训练时一致的小写和去标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def classify(self, text: str) -> int:
        """
        对输入的文本进行谣言检测
        Args:
            text: 输入的文本字符串
        Returns:
            int: 预测的类别（0表示非谣言，1表示谣言）
        """
        text = self.preprocess(text)
        X_vec = self.vectorizer.transform([text])
        pred = self.model.predict(X_vec)[0]
        return int(pred)

# 测试代码
if __name__ == "__main__":
    detector = RumourDetectClass()
    test_texts = [
        "This is a normal tweet.",
        "Breaking news: a major event has happened!",
        "This is a false alarm, nothing to worry about."
    ]

    for text in test_texts:
        result = detector.classify(text)
        print(f"Text: '{text}' -> Prediction: {result}")

conda上代码运行：
conda activate new_env
F:
cd \大学\大二下\人工智能导论（A类）\大作业\我的代码\
注意这里直接cd F:\大学\大二下\人工智能导论（A类）\大作业\代码\是不会进入文件夹位置的，还是在C盘。
 
python train_model.py
但是要记得根据头文件pip install pandas scikit-learn
 
python classify.py
 
很明显效果不行，但跑通了。
自己换了BiGRU 模型后结果：
 
 

同组代码
conda create -n new_env1 python=3.8和conda activate new_env1并conda init新环境。
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
重新conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
开始测试同组代码
conda activate new_env1
F:
cd \大学\大二下\人工智能导论（A类）\大作业\同组代码\
python try.py
但有一些包要装
conda install pandas
conda install scikit-learn
conda install transformers -c conda-forge
发现还是报错，pillow不正常。
conda uninstall pillow
conda install pillow
 
conda install nltk
但huggingface.co现在在国内不可用，预训练模型无法下载，于是：
set HF_ENDPOINT=https://hf-mirror.com
 
但是随之而来的是AttributeError: module 'torch' has no attribute 'frombuffer'也就是PyTorch 版本与 safetensors 库不兼容。我的PyTorch原本是按照老师的1.8（<2.0），但是这里要更新的才行。
conda remove pytorch torchvision torchaudio cudatoolkit和conda clean –all后
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
python -c "import torch; print(torch.__version__)"
 
现在python try.py：
 
此时代码：
try.py:
import re
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

torch.manual_seed(42)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class RumourDetector:
    def __init__(self, bert_model_name='distilbert-base-uncased',  # Using smaller model
                 max_seq_length=64, tfidf_max_features=3000,
                 hidden_dims=[128, 64, 64], dropout_rate=0.3,
                 batch_size=32, learning_rate=2e-4, freeze_bert=True):  # Added freeze_bert
        """
        Initialize the rumour detector with optimized settings

        Args:
            bert_model_name: Name of pretrained BERT model (default: distilbert for efficiency)
            max_seq_length: Maximum sequence length for BERT
            tfidf_max_features: Maximum number of TF-IDF features
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout rate
            batch_size: Training batch size
            learning_rate: Learning rate
            freeze_bert: Whether to freeze BERT weights (reduces training cost)
        """
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            bert_model_name=bert_model_name,
            max_seq_length=max_seq_length,
            tfidf_max_features=tfidf_max_features,
            freeze_bert=freeze_bert
        )

        # Classifier will be initialized during training
        self.classifier = None
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Training state
        self.is_trained = False

    def preprocess_text(self, text):
        """Optimized text preprocessing pipeline"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word)
                 for word in words
                 if word not in self.stop_words and len(word) > 2]  # Added length filter
        return ' '.join(words)

    def train(self,  train_file, val_file, epochs=10):
        """
        Train the model with early stopping and learning rate scheduling

        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts
            val_labels: List of validation labels
            epochs: Maximum number of training epochs
        """

        # Preprocess texts
        train_df = pd.read_csv(r'F:\大学\大二下\人工智能导论（A类）\大作业\谣言数据集\train.csv')
        val_df = pd.read_csv(r'F:\大学\大二下\人工智能导论（A类）\大作业\谣言数据集\val.csv')
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()
        train_texts = [self.preprocess_text(text) for text in train_texts]
        val_texts = [self.preprocess_text(text) for text in val_texts]

        train_labels = train_df['label'].tolist()
        val_labels = val_df['label'].tolist()

        # Extract features
        X_train = self.feature_extractor.fit_transform(train_texts)
        X_val = self.feature_extractor.transform(val_texts)

        # Convert to tensors
        y_train = torch.LongTensor(train_labels)
        y_val = torch.LongTensor(val_labels)

        # Initialize classifier now that we know feature dimensions
        input_dim = X_train.shape[1]
        self.classifier = Classifier(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )

        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)

        # Train with early stopping
        best_acc = 0
        patience = 10
        no_improve = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': self.classifier.parameters()},
            {'params': self.feature_extractor.get_trainable_params(), 'lr': 1e-5}  # Lower LR for BERT
        ], lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss, train_acc = self._run_epoch(train_loader, criterion, optimizer, train=True)

            # Validation
            self.classifier.eval()
            val_loss, val_acc = self._run_epoch(val_loader, criterion, None, train=False)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                self.save_model('best_model')  # Save best model automatically
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step(val_acc)

        self.is_trained = True
        return best_acc

    def _run_epoch(self, loader, criterion, optimizer, train=True):
        """Run one epoch of training or validation"""
        total_loss = 0
        correct = 0
        total = 0

        for X, y in loader:
            if train:
                optimizer.zero_grad()

            outputs = self.classifier(X)
            loss = criterion(outputs, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        return total_loss / len(loader), 100 * correct / total

    def _create_data_loader(self, X, y, shuffle):
        """Create a PyTorch DataLoader from sparse features"""
        X = torch.FloatTensor(X.toarray())
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def predict(self, text):
        """Make prediction on a single text"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Preprocess and extract features
        cleaned_text = self.preprocess_text(text)
        X = self.feature_extractor.transform([cleaned_text])
        X = torch.FloatTensor(X.toarray())

        # Predict
        self.classifier.eval()
        with torch.no_grad():
            output = self.classifier(X)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    def save_model(self, save_dir):
        """Save entire model including feature extractor and classifier"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save classifier
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))

        # Save feature extractor
        self.feature_extractor.save(save_dir)

        # Save model parameters
        params = {
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        with open(os.path.join(save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        print(f"Model saved to {save_dir}")

    @classmethod
    def load_model(cls, save_dir):
        """Load saved model"""
        # Check required files exist
        required_files = ['classifier.pth', 'params.pkl',
                          'tfidf.pkl', 'feature_dimensions.pkl']
        for f in required_files:
            if not os.path.exists(os.path.join(save_dir, f)):
                raise FileNotFoundError(f"Missing required file: {f}")

        # Load parameters
        with open(os.path.join(save_dir, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)

        # Initialize model
        model = cls(
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate']
        )

        # Load feature extractor
        model.feature_extractor = FeatureExtractor.load(save_dir)

        # Initialize and load classifier
        input_dim = model.feature_extractor.feature_dimensions['total']
        model.classifier = Classifier(
            input_dim=input_dim,
            hidden_dims=model.hidden_dims,
            dropout_rate=model.dropout_rate
        )
        model.classifier.load_state_dict(
            torch.load(os.path.join(save_dir, 'classifier.pth')))

        model.is_trained = True
        print(f"Model loaded from {save_dir}")
        return model


class FeatureExtractor:
    def __init__(self, bert_model_name='distilbert-base-uncased',
                 max_seq_length=64, tfidf_max_features=3000, freeze_bert=True):
        """
        Optimized feature extractor with frozen BERT by default

        Args:
            bert_model_name: Name of pretrained model
            max_seq_length: Maximum sequence length
            tfidf_max_features: Number of TF-IDF features
            freeze_bert: Whether to freeze BERT weights
        """
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),  # Reduced from (1,3) for efficiency
            max_features=tfidf_max_features,
            stop_words='english'  # Use built-in stopwords
        )

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.max_seq_length = max_seq_length

        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Track feature dimensions
        self.feature_dimensions = {}

    def fit_transform(self, texts):
        """Fit and transform texts to features"""
        # TF-IDF features
        X_tfidf = self.tfidf.fit_transform(texts)
        self.feature_dimensions['tfidf'] = X_tfidf.shape[1]

        # BERT features
        X_bert = self._get_bert_features(texts)
        self.feature_dimensions['bert'] = X_bert.shape[1]

        # Combine features
        X_combined = hstack([X_tfidf, csr_matrix(X_bert)])
        self.feature_dimensions['total'] = X_combined.shape[1]

        return X_combined

    def transform(self, texts):
        """Transform new texts to features"""
        X_tfidf = self.tfidf.transform(texts)
        X_bert = self._get_bert_features(texts)
        return hstack([X_tfidf, csr_matrix(X_bert)])

    def _get_bert_features(self, texts):
        """Get BERT embeddings efficiently"""
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        # Use mean pooling instead of [CLS] for better stability
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def get_trainable_params(self):
        """Get parameters that require gradients (for optimization)"""
        return [p for p in self.bert.parameters() if p.requires_grad]

    def save(self, save_dir):
        """Save feature extractor components"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save TF-IDF
        with open(os.path.join(save_dir, 'tfidf.pkl'), 'wb') as f:
            pickle.dump(self.tfidf, f)

        # Save BERT (just config to save space)
        self.bert.config.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save feature dimensions
        with open(os.path.join(save_dir, 'feature_dimensions.pkl'), 'wb') as f:
            pickle.dump(self.feature_dimensions, f)

    @classmethod
    def load(cls, save_dir):
        """Load feature extractor"""
        # Initialize
        extractor = cls()

        # Load TF-IDF
        with open(os.path.join(save_dir, 'tfidf.pkl'), 'rb') as f:
            extractor.tfidf = pickle.load(f)

        # Load BERT
        extractor.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        extractor.bert = AutoModel.from_pretrained(save_dir)

        # Load feature dimensions
        with open(os.path.join(save_dir, 'feature_dimensions.pkl'), 'rb') as f:
            extractor.feature_dimensions = pickle.load(f)

        return extractor


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        """Optimized classifier with batch norm and dropout"""
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 初始化模型
detector = RumourDetector()

# 训练并自动保存最佳模型
best_val_acc = detector.train(
    r'F:\大学\大二下\人工智能导论（A类）\大作业\谣言数据集\train.csv',
    r'F:\大学\大二下\人工智能导论（A类）\大作业\谣言数据集\val.csv',
    15
)
调参：
hidden_dims=[256, 128, 64]时：
