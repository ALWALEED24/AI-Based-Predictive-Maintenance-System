import pandas as pd


def load_dataset(file_path):
    """
    Load the AI4I 2020 Predictive Maintenance dataset.
    """
    try:
        data = pd.read_csv(file_path)

        print("Dataset loaded successfully.")
        print(f"Dataset shape: {data.shape}")

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return None