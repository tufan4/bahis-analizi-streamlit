import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import re

DATA_DIR = "feather_data"

# === LOAD FEATHER FILES ===
def load_data():
    feather_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.feather')]
    all_dfs = []
    for file in feather_files:
        try:
            df = pd.read_feather(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"[!] {file} okunamadı: {e}")
    if not all_dfs:
        raise ValueError("Hiçbir valid .feather dosyası bulunamadı.")
    return pd.concat(all_dfs, ignore_index=True)

# === NORMALIZE COL NAMES ===
def normalize_col(col):
    return re.sub(r'[^\w\s]', '', col).strip().lower().replace(' ', '_')

# === FIND TEAM COLUMNS ===
def find_team_columns(df):
    cols = df.columns.tolist()
    ev_col, dep_col, lig_col = None, None, None
    for c in cols:
        nc = normalize_col(c)
        if 'ev_sahibi' in nc or 'home' in nc:
            ev_col = c
        elif 'deplasman' in nc or 'away' in nc:
            dep_col = c
        elif 'lig' in nc or 'league' in nc:
            lig_col = c
    return ev_col, dep_col, lig_col

# === PARSE SCORE ===
def parse_score(score):
    if not isinstance(score, str):
        return None, None
    score = score.strip().replace(',', ':').replace('-', ':').replace(' ', ':')
    parts = [p for p in score.split(':') if p.strip()]
    if len(parts) != 2:
        return None, None
    try:
        home = int(parts[0].strip())
        away = int(parts[1].strip())
        return home, away
    except Exception:
        return None, None

# === SCORE FREQUENCY PLOT ===
def plot_top_scores(scores, title="En Çok Tekrar Eden Skorlar"):
    score_counter = Counter(scores)
    top_scores = score_counter.most_common(10)
    labels = [f"{h}-{a}" for h, a in top_scores]
    counts = [c for (_, _), c in top_scores]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, counts, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Skor")
    ax.set_ylabel("Tekrar Sayısı")
    return fig

# === GOAL DISTRIBUTION PLOT ===
def plot_goal_distribution(goals, title="Toplam Gol Dağılımı"):
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = range(0, 10)
    ax.hist(goals, bins=bins, color='lightgreen', edgecolor='black', rwidth=0.9)
    ax.set_xticks(bins)
    ax.set_title(title)
    return fig

# === MATCH ANALYSIS ===
def analyze_match(user_odds, max_match=500):
    df = load_data()
    ev_col, dep_col, lig_col = find_team_columns(df)

    cols = ['1-A MS', '1-K MS', 'X-A MS', 'X-K MS', '2-A MS', '2-K MS']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['total_closeness'] = 0
    for i, col in enumerate(cols):
        df[f"close_{col}"] = (df[col] - user_odds[i])**2
        df['total_closeness'] += df[f"close_{col}"]

    closest = df.sort_values('total_closeness').head(int(max_match))

    closest['MS_ev_gol'], closest['MS_dep_gol'] = zip(*closest['MS'].map(parse_score))
    closest = closest.dropna(subset=['MS_ev_gol', 'MS_dep_gol'])
    closest['MS_ev_gol'] = closest['MS_ev_gol'].astype(int)
    closest['MS_dep_gol'] = closest['MS_dep_gol'].astype(int)

    scores = [(int(h), int(a)) for h, a in zip(closest['MS_ev_gol'], closest['MS_dep_gol'])]
    total_goals = closest['MS_ev_gol'] + closest['MS_dep_gol']

    fig_skor = plot_top_scores(scores)
    fig_hist = plot_goal_distribution(total_goals)

    ek_sutunlar = [ev_col, dep_col, lig_col] if lig_col else [ev_col, dep_col]
    tablo = closest[ek_sutunlar + ['MS', 'total_closeness']].reset_index(drop=True)

    summary = f"""
📊 En yakın {len(closest)} maçı bulduk.
🟦 Toplam gol ortalaması: {total_goals.mean():.2f}
🟨 Standart sapma: {total_goals.std():.2f}
    """
    return tablo, summary, fig_skor, fig_hist, closest

# === PREDICT MOST LIKELY SCORES ===
def predict_scores(closest):
    ev_goals = closest['MS_ev_gol']
    dep_goals = closest['MS_dep_gol']

    ev_freq = Counter(ev_goals)
    dep_freq = Counter(dep_goals)

    ev_top = ev_freq.most_common()
    dep_top = dep_freq.most_common()

    combined_scores = []
    for e, _ in ev_top:
        for d, _ in dep_top:
            combined_scores.append((e, d))

    combined_scores.sort(key=lambda x: ev_freq.get(x[0], 0) + dep_freq.get(x[1], 0), reverse=True)
    return combined_scores[:5]

# === STREAMLIT UI ===
st.set_page_config(page_title="BLACKFOOT-XL", layout="wide")
st.title("⚽ BLACKFOOT-XL – Karanlık Yapay Zekalı Futbol Tahminci")

st.subheader("Oranları Giriniz")
oran_1a = st.number_input("1-A MS", value=1.0, step=0.01)
oran_1k = st.number_input("1-K MS", value=1.0, step=0.01)
oran_xa = st.number_input("X-A MS", value=1.0, step=0.01)
oran_xk = st.number_input("X-K MS", value=1.0, step=0.01)
oran_2a = st.number_input("2-A MS", value=1.0, step=0.01)
oran_2k = st.number_input("2-K MS", value=1.0, step=0.01)

user_odds = [oran_1a, oran_1k, oran_xa, oran_xk, oran_2a, oran_2k]

if st.button("Analiz Et"):
    tablo, summary, fig_skor, fig_hist, closest = analyze_match(user_odds)
    st.session_state["closest"] = closest  # session'a kaydet
    st.subheader("Sonuç Tablosu")
    st.dataframe(tablo)
    st.subheader("Genel Analiz")
    st.text(summary)
    st.subheader("En Çok Tekrar Eden Skorlar")
    st.pyplot(fig_skor)
    st.subheader("Toplam Gol Histogramı")
    st.pyplot(fig_hist)

if "closest" in st.session_state and st.button("En Olası 5 Kesin Skoru Hesapla"):
    closest = st.session_state["closest"]
    st.subheader("En Olası 5 Kesin Skor")
    top_scores = predict_scores(closest)
    for i, (ev, dep) in enumerate(top_scores, 1):
        st.write(f"{i}. {ev}-{dep}")
