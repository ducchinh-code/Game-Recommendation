import pandas as pd

df_long =  pd.read_csv("data/user_game_long.csv")

df_wide = df_long.pivot_table(index="user_id", columns="game", values="rating", fill_value="")
df_wide.reset_index(inplace=True)
df_wide.to_csv("user_game_wide.csv", index=False, encoding="utf-8")

print("✅ Đã chuyển từ long -> wide và lưu vào 'user_game_wide.csv'")