import pandas as pd

df = pd.read_csv("dataset_fitur.csv")

df_fall = df[df['label'] == 'fall']
df_normal = df[df['label'] == 'normal']

print(f"Sebelum balance:")
print(f"  Fall: {len(df_fall)}")
print(f"  Normal: {len(df_normal)}")

#ambil sample dari df_normal sebanyak 2x df_fall
df_normal_balanced = df_normal.sample(n=len(df_fall) * 2, random_state=42)

df_final = pd.concat([df_fall, df_normal_balanced])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

df_final.to_csv("dataset_fitur_balanced.csv", index=False)

print(f"\nSetelah balance:")
print(f"  Fall: {len(df_final[df_final['label']=='fall'])}")
print(f"  Normal: {len(df_final[df_final['label']=='normal'])}")
print(f"  Total: {len(df_final)}")