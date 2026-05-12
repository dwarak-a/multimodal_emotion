import pandas as pd
import whisper
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def generate_transcripts(input_csv=r'..\processed\metadata.csv', 
                         output_csv=r'..\processed\metadata_with_text.csv'):
    
    print("Loading existing metadata...")
    df = pd.read_csv(input_csv)
    
    print("Loading Whisper 'base' model... (this may take a moment)")
    model = whisper.load_model("base")
    
    transcripts = []
    
    print(f"Transcribing {len(df)} audio files...")
    for file_path in tqdm(df['file_path']):
        if os.path.exists(file_path):
            result = model.transcribe(file_path)
            transcripts.append(result["text"].strip())
        else:
            print(f"File not found: {file_path}")
            transcripts.append("")
            
    df['transcript'] = transcripts
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Transcripts saved to '{output_csv}'")


if __name__ == "__main__":
    generate_transcripts()