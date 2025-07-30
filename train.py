import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
class MultimodalClassifier(pl.LightningModule):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=[3, 2, 2, 2], lr=2e-5):
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

        self.criterions = [nn.CrossEntropyLoss() for _ in range(len(num_classes))]
        self.lr = lr

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(text_out.pooler_output)

        img_feat = self.image_encoder(image).squeeze()
        img_feat = self.image_fc(img_feat)

        combined = torch.cat([text_feat, img_feat], dim=1)
        combined = self.dropout(self.fusion_fc(combined))

        outputs = [clf(combined) for clf in self.classifiers]
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        images = inputs['image']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = self(images, input_ids, attention_mask)
        loss = sum(self.criterions[i](outputs[i], labels[:, i]) for i in range(len(self.classifiers)))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ------------------------------
# Training with PyTorch Lightning
# ------------------------------
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "Bangla_train_2025/Bangla_train_data.csv"
    IMAGE_FOLDER = "Bangla_train_2025/Bangla_train_images"
    BATCH_SIZE = 16
    LR = 2e-5
    MAX_EPOCHS = 50

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

    model = MultimodalClassifier(lr=LR)

    # EarlyStopping Callback
    early_stopping = EarlyStopping(
        monitor='train_loss',
        patience=5,
        mode='min'
    )

    # ModelCheckpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    # Train the model
    trainer.fit(model, train_loader)


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train()