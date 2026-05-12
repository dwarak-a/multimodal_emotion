import torch
import torchaudio
import pandas as pd
import soundfile as sf
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
from transformers import DistilBertTokenizer


# TEXT DATASET

class RavdessTextDataset(Dataset):
    def __init__(self, metadata_path, max_length=32):
        self.metadata = pd.read_csv(metadata_path).fillna('')
        self.max_length = max_length
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        transcript = str(row['transcript']) 
        label = self.emotions.index(row['emotion'])
        
        encoding = self.tokenizer(
            transcript,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
            
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)



# AUDIO DATASET

class RavdessAudioDataset(Dataset):
    def __init__(self, metadata_path, target_sample_rate=22050, max_length=200):
        self.metadata = pd.read_csv(metadata_path)
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.mel_spectrogram = MelSpectrogram(sample_rate=target_sample_rate, n_mels=128, n_fft=1024, hop_length=512)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.resample = torchaudio.transforms.Resample(48000, target_sample_rate)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        label = self.emotions.index(row['emotion'])
        waveform, sample_rate = sf.read(row['file_path'])
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.transpose(0, 1)
        
        if sample_rate != self.target_sample_rate:
            waveform = self.resample(waveform)
            
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        spectrogram = self.mel_spectrogram(waveform)
        curr_length = spectrogram.shape[2]
        
        if (curr_length > self.max_length): spectrogram = spectrogram[:, :, :self.max_length]
        else: spectrogram = F.pad(spectrogram, (0, self.max_length - curr_length))
        
        spectrogram = self.amplitude_to_db(spectrogram)
        
        return spectrogram, torch.tensor(label, dtype=torch.long)



# MULTIMODAL DATASET

class RavdessMultimodalDataset(Dataset):
    def __init__(self, metadata_path):
        self.audio_ds = RavdessAudioDataset(metadata_path)
        self.text_ds = RavdessTextDataset(metadata_path)
        self.metadata = self.audio_ds.metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        spectrogram, label = self.audio_ds[index]
        input_ids, attention_mask, _ = self.text_ds[index]
        
        return spectrogram, input_ids, attention_mask, label