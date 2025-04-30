import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# 1. Load the original CSV
df = pd.read_csv("Spotify_final_dataset.csv")


# Identify and print rows with missing Song Name
missing_rows = df[df["Song Name"].isnull()]
print("Rows with missing Song Name:")
print(missing_rows)

# Fill missing Song Name values
df["Song Name"] = df["Song Name"].fillna("Unknown title")

"""# 4. Convert Top10(xTimes) from float to int
df["Top 10 (xTimes)"] = df["Top 10 (xTimes)"].astype(int)

# 5. Parse Peak Position (xTimes) into an integer column
df["Peak Position (xTimes)"] = (
    df["Peak Position (xTimes)"]
      .str.extract(r"\(x(\d+)\)")  # capture the number inside "(x…)"
      .fillna("0")                 # replace any non‐matches with "0"
      .astype(int)                 # convert to integer dtype
)

# 6. (Optional) Reset the DataFrame index
df = df.reset_index(drop=True)

# 7. Save the cleaned DataFrame to a new CSV
#df.to_csv("Spotify_cleaned.csv", index=False)

print("✅ Cleaning complete. Saved to Spotify_cleaned.csv")"""