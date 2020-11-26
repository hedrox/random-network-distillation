import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model1_path", type=str, help="Path to the progress.csv of the first")
parser.add_argument("model2_path", type=str, help="Path to the progress.csv of the second")
args = parser.parse_args()

data_csv1 = pd.read_csv(args.model1_path)
data_csv2 = pd.read_csv(args.model2_path)

res = data_csv1.copy()

base_reward = data_csv1.rewtotal.values[-1]
base_tcount = data_csv1.tcount.values[-1]
visited_rooms = set()
highest_reward = data_csv1.best_ret.values[-1]

def format_rooms(rooms):
    # Input rooms are strings: '[1;2;3]'
    rooms = rooms.strip("][")
    if rooms:
        rooms = rooms.split(";")
        return rooms
    return []

rooms = format_rooms(data_csv1.rooms.values[-1])
for r in rooms:
    visited_rooms.add(r)

for row in data_csv2.values:
    reward = row[43]
    if pd.isna(reward):
        continue

    tcount = row[32]
    rooms = format_rooms(row[31])
    for r in rooms:
        visited_rooms.add(r)
    highest_reward = max(row[2], highest_reward)

    total_reward = base_reward + reward
    total_tcount = base_tcount + tcount
    frame = list(row)
    frame[2] = highest_reward
    frame[8] = len(visited_rooms)    
    frame[43] = total_reward
    frame[32] = total_tcount
    df = pd.DataFrame([frame], columns=list(data_csv1.columns))
    res = res.append(df, ignore_index=True)

res.to_csv("aggregated_progress.csv")

    
