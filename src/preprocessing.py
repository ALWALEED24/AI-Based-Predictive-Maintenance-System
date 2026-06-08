import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    """
    Preprocess the AI4I 2020 Predictive Maintenance dataset.
    """

    if data is None:
        print("No data found for preprocessing.")
        return None, None, None

    df = data.copy()

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Drop columns that are not useful for modelling
    columns_to_drop = ["UDI", "Product ID"]

    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Encode Type column: L, M, H
    if "Type" in df.columns:
        type_mapping = {
            "L": 0,
            "M": 1,
            "H": 2
        }
        df["Type"] = df["Type"].map(type_mapping)

    # Input features
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # Target columns for evaluation
    target_columns = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    # Check if required columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]

    if missing_features:
        print(f"Missing feature columns: {missing_features}")
        return None, None, None

    if missing_targets:
        print(f"Missing target columns: {missing_targets}")
        return None, None, None

    # Remove missing values
    df.dropna(inplace=True)

    # Separate features and targets
    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    # Scale the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    # Data before scaling
    cleaned_data = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )

    # Data after scaling
    scaled_data = pd.concat(
        [X_scaled_df.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )

    print("\nPreprocessing completed successfully.")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Scaled data shape: {scaled_data.shape}")

    return cleaned_data, scaled_data, scaler


def save_preprocessed_data(cleaned_data, scaled_data, output_folder="outputs"):
    """
    Save cleaned and scaled data into the outputs folder.
    """

    if cleaned_data is None or scaled_data is None:
        print("No preprocessed data to save.")
        return

    os.makedirs(output_folder, exist_ok=True)

    cleaned_path = os.path.join(output_folder, "cleaned_data.csv")
    scaled_path = os.path.join(output_folder, "scaled_data.csv")

    cleaned_data.to_csv(cleaned_path, index=False)
    scaled_data.to_csv(scaled_path, index=False)

    print(f"\nCleaned data saved to: {cleaned_path}")
    print(f"Scaled data saved to: {scaled_path}")