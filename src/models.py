import torch
import torch.nn as nn
from transformers import DistilBertModel


# AUDIO MODEL

class AudioCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.classifier = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        return self.classifier(x)
    
    def get_bottleneck_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return self.fc2(x)


# TEXT MODEL

class TextDistilBERT(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Sequential(
            nn.Linear(768, 128), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        x = self.fc2(self.fc1(cls_token_state))
        return self.output_layer(x)
    
    def get_bottleneck_features(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        return self.fc2(self.fc1(cls_token_state))


# MULTIMODAL MODEL

class MultimodalEarlyFusion(nn.Module):
    def __init__(self, audio_model, text_model, num_classes=8):
        super().__init__()
        self.audio_model = audio_model
        self.text_model = text_model

        # Updated to accept 128 from audio and 64 from text
        self.fusion_classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_x, input_ids, attention_mask):
        audio_features = self.audio_model.get_bottleneck_features(audio_x)
        text_features = self.text_model.get_bottleneck_features(input_ids, attention_mask)

        fused_features = torch.cat((audio_features, text_features), dim=1)
        return self.fusion_classifier(fused_features)