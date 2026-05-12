import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datasets import RavdessAudioDataset, RavdessTextDataset, RavdessMultimodalDataset
from models import AudioCNN, TextDistilBERT, MultimodalEarlyFusion

def plot_losses(train_losses, val_losses, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# AUDIO TRAINING
# ==========================================
def train_audio(batch_size=16, epochs=50, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = RavdessAudioDataset(r'..\processed\metadata.csv')
    df = full_dataset.metadata
    
    train_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] <= 20].index.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] > 20].index.tolist()), batch_size=batch_size, shuffle=False)
    
    model = AudioCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    history_train_loss, history_val_loss, best_val_loss = [], [], float('inf')
        
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.size(0)
            inputs_flat = inputs.view(bs, -1)
            inputs = (inputs - inputs_flat.mean(dim=1).view(bs, 1, 1, 1)) / (inputs_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
            
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                bs = inputs.size(0)
                inputs_flat = inputs.view(bs, -1)
                inputs = (inputs - inputs_flat.mean(dim=1).view(bs, 1, 1, 1)) / (inputs_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {100 * correct / total:.2f}%")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(r'..\processed', exist_ok=True)
            torch.save(model.state_dict(), r'..\processed\audio_cnn_weights.pth')
            
    plot_losses(history_train_loss, history_val_loss, r'..\processed\audio_loss_plot.png', 'Audio CNN')

    # Final Evaluation
    model.load_state_dict(torch.load(r'..\processed\audio_cnn_weights.pth', map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.size(0)
            inputs_flat = inputs.view(bs, -1)
            inputs = (inputs - inputs_flat.mean(dim=1).view(bs, 1, 1, 1)) / (inputs_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
            outputs = model(inputs)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n" + "="*30)
    print("FINAL AUDIO CNN METRICS (BEST MODEL)")
    print("="*30)
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("="*30 + "\n")

# ==========================================
# TEXT TRAINING
# ==========================================
def train_text(batch_size=16, epochs=30, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Frozen DistilBERT Training on {device} ---")
    
    full_dataset = RavdessTextDataset(r'..\processed\metadata_with_text.csv')
    df = full_dataset.metadata
    train_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] <= 20].index.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] > 20].index.tolist()), batch_size=batch_size, shuffle=False)
    
    model = TextDistilBERT().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    history_train_loss, history_val_loss, best_val_loss = [], [], float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {100 * correct / total:.2f}%")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(r'..\processed', exist_ok=True)
            torch.save(model.state_dict(), r'..\processed\text_distilbert_weights.pth')
            
    plot_losses(history_train_loss, history_val_loss, r'..\processed\text_loss_plot.png', 'Text DistilBERT')

    # Final Evaluation
    model.load_state_dict(torch.load(r'..\processed\text_distilbert_weights.pth', map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n" + "="*30)
    print("FINAL TEXT BERT METRICS (BEST MODEL)")
    print("="*30)
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("="*30 + "\n")

# ==========================================
# MULTIMODAL TRAINING
# ==========================================
def train_multimodal(batch_size=16, epochs=30, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Multimodal Early Fusion on {device} ---")
    
    full_dataset = RavdessMultimodalDataset(r'..\processed\metadata_with_text.csv')
    df = full_dataset.metadata
    
    train_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] <= 20].index.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, df[df['actor_id'] > 20].index.tolist()), batch_size=batch_size, shuffle=False)
    
    audio_model, text_model = AudioCNN(), TextDistilBERT()
    
    if os.path.exists(r'..\processed\audio_cnn_weights.pth'):
        audio_model.load_state_dict(torch.load(r'..\processed\audio_cnn_weights.pth', map_location=device, weights_only=True))
    if os.path.exists(r'..\processed\text_distilbert_weights.pth'):
        text_model.load_state_dict(torch.load(r'..\processed\text_distilbert_weights.pth', map_location=device, weights_only=True))
        
    model = MultimodalEarlyFusion(audio_model, text_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    history_train_loss, history_val_loss, best_val_loss = [], [], float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for audio_x, input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False):
            audio_x, input_ids, attention_mask, labels = audio_x.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            bs = audio_x.size(0)
            audio_flat = audio_x.view(bs, -1)
            audio_x = (audio_x - audio_flat.mean(dim=1).view(bs, 1, 1, 1)) / (audio_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
            
            optimizer.zero_grad()
            loss = criterion(model(audio_x, input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for audio_x, input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False):
                audio_x, input_ids, attention_mask, labels = audio_x.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                bs = audio_x.size(0)
                audio_flat = audio_x.view(bs, -1)
                audio_x = (audio_x - audio_flat.mean(dim=1).view(bs, 1, 1, 1)) / (audio_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
                
                outputs = model(audio_x, input_ids, attention_mask)
                val_loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {100 * correct / total:.2f}%")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(r'..\processed', exist_ok=True)
            torch.save(model.state_dict(), r'..\processed\multimodal_weights.pth')
            
    plot_losses(history_train_loss, history_val_loss, r'..\processed\multimodal_loss_plot.png', 'Multimodal Fusion')

    # Final Evaluation
    model.load_state_dict(torch.load(r'..\processed\multimodal_weights.pth', map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio_x, input_ids, attention_mask, labels in val_loader:
            audio_x, input_ids, attention_mask, labels = audio_x.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            bs = audio_x.size(0)
            audio_flat = audio_x.view(bs, -1)
            audio_x = (audio_x - audio_flat.mean(dim=1).view(bs, 1, 1, 1)) / (audio_flat.std(dim=1).view(bs, 1, 1, 1) + 1e-6)
            
            outputs = model(audio_x, input_ids, attention_mask)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n" + "="*30)
    print("FINAL MULTIMODAL METRICS (BEST MODEL)")
    print("="*30)
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['audio', 'text', 'multimodal'], required=True)
    args = parser.parse_args()
    
    if args.mode == 'audio': train_audio()
    elif args.mode == 'text': train_text()
    elif args.mode == 'multimodal': train_multimodal()