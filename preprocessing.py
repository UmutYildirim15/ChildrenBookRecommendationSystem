
import json
import pandas as pd
import numpy as np

def convert_json_to_data(json_path: str, data_path: str):
    """
    Read each line of the JSON file and write only the 'user_id', 'book_id', and 'rating' fields
    into a whitespace-separated .data file.
    """
    with open(json_path, 'r') as fin, open(data_path, 'w') as fout:
        for line in fin:
            record = json.loads(line)
            user = record.get('user_id')
            book = record.get('book_id')
            rating = record.get('rating', 0)
            # For each record, write: "user_id book_id rating\n"
            fout.write(f"{user} {book} {rating}\n")
    print(f"JSON ➔ {data_path} converted (only user_id, book_id, rating).")


def load_data(data_path: str) -> pd.DataFrame:
    """
    Read the whitespace-delimited .data file into a pandas DataFrame.
    Set column names to user_id, item_id, and rating.
    """
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=["user_id", "item_id", "rating"],
        dtype={"user_id": str, "item_id": str, "rating": int},
        engine="python",
        on_bad_lines="skip"
    )
    return df


def filter_users_items(df: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 20) -> pd.DataFrame:
    """
    Filter out users with fewer than min_user_interactions and
    books with fewer than min_item_interactions.
    """
    # Count interactions per user
    user_counts = df["user_id"].value_counts()
    # Count interactions per book (item)
    item_counts = df["item_id"].value_counts()

    # Identify users and books that meet the minimum interaction thresholds
    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index

    # Keep only rows where both user_id and item_id are in the valid lists
    df_filtered = df[
        df["user_id"].isin(valid_users) &
        df["item_id"].isin(valid_items)
    ].reset_index(drop=True)

    return df_filtered


def downsample_dataframe(df: pd.DataFrame, fraction: float = 0.2) -> pd.DataFrame:
    """
    Randomly keep the specified fraction (e.g., 0.2) of the filtered DataFrame.
    """
    np.random.seed(42)  # Fix seed for reproducibility
    mask = np.random.rand(len(df)) < fraction
    df_down = df[mask].reset_index(drop=True)
    print(f"Randomly sampled {fraction*100:.0f}%: {len(df_down)} rows.")
    return df_down


if __name__ == "__main__":

    json_input_path = "/Users/umutyildirim/Downloads/data/goodreads_interactions_children.json"    # Path to json data
    raw_data_path   = "raw.data"      # Temporary file to hold “user book rating” output
    convert_json_to_data(json_input_path, raw_data_path)

    raw_df = load_data(raw_data_path)

    # Filter: users with ≥ 5 interactions and books with ≥ 20 interactions
    df_filtered = filter_users_items(raw_df, min_user_interactions=5, min_item_interactions=20)

    filtered_data_path = "filtered.data"
    df_filtered.to_csv(filtered_data_path, sep=" ", header=False, index=False)
    print(f"Filtered data saved → {filtered_data_path}")

    df_downsampled = downsample_dataframe(df_filtered, fraction=0.2)

    # Save the final sampled dataset
    final_data_path = "test.data"
    df_downsampled.to_csv(final_data_path, sep=" ", header=False, index=False)
    print(f"Final 20% sample saved → {final_data_path}")

