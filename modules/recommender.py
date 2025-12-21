# modules/recommender.py
import random

health_goals = {
    "减肥": ["低卡", "低脂"],
    "增肌": ["高蛋白"],
    "清淡": ["清淡", "低脂"],
    "辣": ["辣"],
    "荤": ["荤"],
    "素": ["素"]
}

def recommend_dishes(df, keywords, budget=None):
    matched = set()
    for _, row in df.iterrows():
        if any(k in row["菜名"] or k in row["标签"] for k in keywords):
            matched.add(row["菜名"])
    for goal, tags in health_goals.items():
        if goal in keywords:
            for _, row in df.iterrows():
                if any(tag in row["标签"] for tag in tags):
                    matched.add(row["菜名"])
    if not matched:
        return df.sample(3)
    filtered = df[df["菜名"].isin(matched)]
    if budget:
        low, high = budget
        within = filtered[(filtered["价格(元)"] >= low) & (filtered["价格(元)"] <= high)]
        if not within.empty:
            return within.head(3)
        else:
            return df.iloc[(df["价格(元)"] - (low + high)/2).abs().argsort()[:3]]
    return filtered.head(3)
