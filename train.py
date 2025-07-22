import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# ------------------------------
# Dataset
# ------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_folder, tokenizer, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
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

        self.df = self.df.dropna(subset=['Ids', 'OCR'])
        self.df = self.df[self.df['Ids'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['Ids']
        if not image_id.endswith(('.jpg', '.png')):
            image_id += '.jpg'

        image_path = os.path.join(self.image_folder, image_id)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        text = row['OCR']
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
            'attention_mask': encoding['attention_mask'].squeeze(0)
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


# ------------------------------
# Model
# ------------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=[3, 2, 2, 2]):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, 256)

        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(512, 256)

        self.fusion_fc = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)

        self.classifiers = nn.ModuleList([
            nn.Linear(256, c) for c in num_classes
        ])

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(text_out.pooler_output)

        img_feat = self.image_encoder(image).squeeze()
        img_feat = self.image_fc(img_feat)

        combined = torch.cat([text_feat, img_feat], dim=1)
        combined = self.dropout(self.fusion_fc(combined))

        outputs = [clf(combined) for clf in self.classifiers]
        return outputs


# ------------------------------
# Training Loop
# ------------------------------
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "Bangla_train_2025/Bangla_train_data.csv"
    IMAGE_FOLDER = "Bangla_train_2025/Bangla_train_images"
    MODEL_SAVE_PATH = "multimodal_model.pth"

    EPOCHS = 1
    BATCH_SIZE = 16
    LR = 2e-5

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = MultimodalDataset(
        csv_path=CSV_PATH,
        image_folder=IMAGE_FOLDER,
        tokenizer=tokenizer,
        transform=transform,
        is_train=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalClassifier().to(DEVICE)

    criterions = [
        nn.CrossEntropyLoss() for _ in range(4)
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = batch
            images = inputs['image'].to(DEVICE)
            input_ids = inputs['input_ids'].to(DEVICE)
            attention_mask = inputs['attention_mask'].to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)

            loss = sum(criterions[i](outputs[i], labels[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train()
