import pandas as pd
import random

def combine_and_shuffle_csv(file1, file2, output_file):
    # Read the two CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine the two DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle the DataFrame rows
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Save the shuffled DataFrame to a new CSV file
    shuffled_df.to_csv(output_file, index=False)
    print(f"Combined and shuffled data saved to '{output_file}'")

# Replace 'file1.csv' and 'file2.csv' with your actual file paths
combine_and_shuffle_csv('/home/hunter.ma/SAM2-ChannelBased_public/data/LV_train.csv', '/home/hunter.ma/SAM2-ChannelBased_public/data/LA_train.csv', '/home/hunter.ma/SAM2-ChannelBased_public/data/train.csv')
