import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from transformers import BertTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_folder, tokenizer, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path, sep=',')
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_train = is_train
        self.label_maps = {
            'Sentiment': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
            'Sarcasm': {'Non-Sarcastic': 0, 'Sarcastic': 1},
            'Vulgar': {'Non Vulgar': 0, 'Vulgar': 1},
            'Abuse': {'Non-abusive': 0, 'Abusive': 1}
        }

    def __len__(self):
        print(self.df.columns)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['Ids'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Ensure text is a valid string
        text = row['OCR']
        if not isinstance(text, str):
            text = ""  # Replace invalid or missing text with an empty string

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        inputs = {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': row['Ids']
        }

        if self.is_train:
            labels = torch.tensor([
                self.label_maps['Sentiment'][row['Sentiment']],
                self.label_maps['Sarcasm'][row['Sarcasm']],
                self.label_maps['Vulgar'][row['Vulgar']],
                self.label_maps['Abuse'][row['Abuse']]
            ], dtype=torch.long)
            return inputs, labels

        return inputs
    


######################
import torch.nn as nn
from torchvision.models import resnet18
from transformers import BertModel

class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased'):
        super().__init__()
        # Text encoder
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(768, 256)

        # Image encoder
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        # Combined classifier for each task
        self.combined_fc = nn.Linear(512, 256)
        self.heads = nn.ModuleList([
            nn.Linear(256, 3),  # Sentiment (3 classes)
            nn.Linear(256, 2),  # Sarcasm
            nn.Linear(256, 2),  # Vulgar
            nn.Linear(256, 2),  # Abuse
        ])

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.cnn(image)  # [B, 256]
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, 768]
        text_feat = self.text_fc(text_feat)  # [B, 256]

        combined = torch.cat([img_feat, text_feat], dim=1)  # [B, 512]
        x = self.combined_fc(combined)  # [B, 256]
        outputs = [head(x) for head in self.heads]
        return outputs

###################
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn.functional as F
import torch

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        image = inputs['image'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(image, input_ids, attention_mask)

        loss = 0
        for i in range(4):  # One loss per task
            loss += F.cross_entropy(outputs[i], labels[:, i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


##################################
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs in dataloader:
            image = inputs['image'].to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(image, input_ids, attention_mask)
            preds = [torch.argmax(o, dim=1).cpu().numpy() for o in outputs]
            preds = list(zip(*preds))  # Transpose
            all_preds.extend(preds)
            all_ids.extend(inputs['id'])

    return all_ids, all_preds

def save_submission(ids, preds, output_file='submission.csv'):
    df = pd.DataFrame(preds, columns=['Sentiment', 'Sarcasm', 'Vulgar', 'Abuse'])
    df.insert(0, 'Ids', ids)
    df.to_csv(output_file, index=False)






########################
from transformers import BertTokenizer
from torchvision import transforms

# Config
csv_path = 'Bangla_train_2025/Bangla_train_data.csv'         # Your train CSV/TSV
test_csv = 'Bangla_test_2025/bengali_test_data_wo_label.csv'          # For submission
image_folder = 'Bangla_train_2025/Bangla_train_images/'       # Folder with images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 16

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets and loaders
train_dataset = MultimodalDataset(csv_path, image_folder, tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MultimodalDataset(test_csv, image_folder, tokenizer, transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model & optimizer
model = MultimodalClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Train
for epoch in range(5):
    loss = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

# Predict
ids, preds = predict(model, test_loader, device)

# Save CSV
save_submission(ids, preds)
