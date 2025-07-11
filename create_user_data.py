import pandas as pd
import random
import numpy as np

game_df = pd.read_csv("data/dataset.csv")

filtered_games = game_df[
    (game_df["positive_ratings"] + game_df["negative_ratings"] >= 100) &
    (game_df["average_playtime"] > 0) &
    (game_df["median_playtime"] > 0)
].reset_index(drop=True)

num_users = 100000
user_ids = [f"user_{i}" for i in range(num_users)]

user_data = []

for user in user_ids:
    rated_games = filtered_games.sample(random.randint(5, 30))

    for _, game in rated_games.iterrows():
        game_id = game["name"]
        avg_pt = game["average_playtime"]
        median_pt = game["median_playtime"]
        max_ach = game["achievements"]

        playtime = round(random.uniform(0.1, avg_pt * 1.7), 2)

        player_ach = random.randint(0, int(max_ach)) if max_ach > 0 else 0

        p_positive = 0.5  # mặc định: 50% positive

        if playtime > median_pt * 1.8:
            p_positive += 0.2  # ưu tiên positive
        elif playtime > median_pt:
            p_positive -= 0.2  # ưu tiên negative

        p_positive = max(0, min(1, p_positive))

        rated = 1 if random.random() < p_positive else 0

        user_data.append({
            "user_id": user,
            "game_name": game_id,
            "rated": rated,
            "playtime": playtime,
            "player_achievement": player_ach
        })

# Kết quả
user_df = pd.DataFrame(user_data)
file_path = "data/user_data.csv"
user_df.to_csv(file_path, index=False)