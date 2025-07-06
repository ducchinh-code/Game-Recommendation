import pandas as pd
import random
import os
import csv

df = pd.read_csv("data/dataset.csv")
rated_candidates = df[(df["positive_ratings"] + df["negative_ratings"]) > 1000]["name"].tolist()
total_users = 36000
output_file = "data/user_game_long.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "game", "rating"])  # header

    for i in range(total_users):
        user_id = f"U{i+1:06d}"
        num_games = random.randint(1, min(50, len(rated_candidates)))
        rated_games = random.sample(rated_candidates, num_games)

        for game in rated_games:
            row = df[df["name"] == game].iloc[0]
            pos, neg = int(row["positive_ratings"]), int(row["negative_ratings"])
            total = pos + neg
            prob_positive = pos / total
            rating = 1 if random.random() < prob_positive else 0

            writer.writerow([user_id, game, rating])

print(f"✅ Đã tạo file {output_file} dạng dài tại:\n{os.path.abspath(output_file)}")