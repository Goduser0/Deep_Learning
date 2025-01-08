import pandas as pd

df = pd.read_csv("My_TAOD/dataset/DeepPCB_Crop/10-shot/train/0.csv")
print(df)
# df.to_csv("origin.csv")

length = len(df)
num_expand = 99

df_combined = df
for _ in range((num_expand // length) - 1):
    df_combined = pd.concat([df_combined, df], axis=0, ignore_index=False)

df_combined = pd.concat([df_combined, df.head(num_expand%length)], axis=0, ignore_index=False)
# df_combined.to_csv("result.csv")