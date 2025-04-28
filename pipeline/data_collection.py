
import pandas as pd
import os


# Data Loading...
df = pd.read_csv(r"D:\tiny-llm-colab-train\src\2_data_collection.py", sep="\t", header=None, names=["Label", "Text"])


# Data balancing...
def create_balanced_dataset(df):
    
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combine ham "subset" with "spam"
    balanced_data = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_data

balanced_data = create_balanced_dataset(df)


# Encode the Label (spam: 1, ham: 0) [Label Encoding]
balanced_data["Label"] = balanced_data["Label"].map({"ham": 0, "spam": 1})


# Split the data into train, test, and validation set
def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_data = df[:train_end]
    validation_data = df[train_end:validation_end]
    test_data = df[validation_end:]

    return train_data, validation_data, test_data

train_data, validation_data, test_data = random_split(balanced_data, 0.7, 0.1)
# Test size is implied to be 0.2 as the remainder


# Save the data into dada/raw folder...
data_path = os.path.join("data", "raw")
os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
validation_data.to_csv(os.path.join(data_path, "validation.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
