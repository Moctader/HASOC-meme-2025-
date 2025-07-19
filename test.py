import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# ------------------------------
# Multimodal Dataset Definition
# ------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_folder, tokenizer, transform=None, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_train = is_train

        # Clean missing
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
            print(f"[WARNING] Image not found: {image_path}, using blank image.")
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
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': row['Ids']
        }

        return inputs


# ------------------------------
# Multimodal Model Definition
# ------------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=[3, 2, 2, 2]):
        super(MultimodalClassifier, self).__init__()
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
# Main Prediction Logic
# ------------------------------
def predict():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "multimodal_model.pth"
    TEST_CSV = "Bangla_test_2025/bengali_test_data_wo_label.csv"
    IMAGE_FOLDER = "Bangla_test_2025/Bangla_test_images/"
    SUBMISSION_CSV = "submission.csv"

    # Label maps (reverse)
    inv_label_maps = {
        'Sentiment': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        'Sarcasm': {0: 'Non-Sarcastic', 1: 'Sarcastic'},
        'Vulgar': {0: 'Non Vulgar', 1: 'Vulgar'},
        'Abuse': {0: 'Non-abusive', 1: 'Abusive'}
    }

    # Load tokenizer & transforms
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load model
    model = MultimodalClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load dataset
    test_dataset = MultimodalDataset(
        csv_path=TEST_CSV,
        image_folder=IMAGE_FOLDER,
        tokenizer=tokenizer,
        transform=transform,
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Inference loop
    submission_rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            ids = batch['id']

            outputs = model(images, input_ids, attention_mask)
            preds = [torch.argmax(o, dim=1).cpu().tolist() for o in outputs]

            for i in range(len(ids)):
                row = {
                    'Ids': ids[i],
                    'Sentiment': inv_label_maps['Sentiment'][preds[0][i]],
                    'Sarcasm': inv_label_maps['Sarcasm'][preds[1][i]],
                    'Vulgar': inv_label_maps['Vulgar'][preds[2][i]],
                    'Abuse': inv_label_maps['Abuse'][preds[3][i]]
                }
                submission_rows.append(row)

    # Save to CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df[['Ids', 'Sentiment', 'Sarcasm', 'Vulgar', 'Abuse']]
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"✅ Submission file saved as {SUBMISSION_CSV}")


# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    predict()
