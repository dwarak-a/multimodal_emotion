import os
import pandas as pd


def build_metadata(data_dir):
    file_data = []
    
    emotions = [
        'neutral', 'calm', 'happy', 'sad',
        'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                name_parts = file.split('.')[0].split('-')
                
                emotion = emotions[int(name_parts[2]) - 1]
                actor_id = int(name_parts[6])
                
                file_data.append({
                    'file_path' : file_path,
                    'emotion' : emotion,
                    'actor_id' : actor_id
                })
                
    df = pd.DataFrame(file_data)
    df_path = r'..\processed\metadata.csv'
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    df.to_csv(df_path, index=False)
    print(f'Metadata ({len(df)} entries) stored in "{df_path}"')
                
                
if __name__ == "__main__":
    data_dir = r'..\data'
    df = build_metadata(data_dir)