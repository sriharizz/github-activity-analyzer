# full safe app.py — uses SHAP if available, otherwise shows fallback contributions
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt

st.set_page_config(page_title="GitHub Activity Signal", layout="wide")
st.title("🐙 GitHub Activity Signal — Developer Activeness Analyzer")

MODEL_PATH = "activity_best_model.joblib"
LE_PATH = "label_encoder.joblib"

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le

model, label_enc = load_model_and_encoder()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

def get_user(username):
    url = f"https://api.github.com/users/{username}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    return r.json() if r.status_code == 200 else None

def get_repos(username):
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page":100, "page":page, "type":"owner", "sort":"pushed"}
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        repos.extend(data)
        if len(data) < 100:
            break
        page += 1
    return repos

def extract_features(user, repos):
    now = datetime.now(timezone.utc)
    created = pd.to_datetime(user.get("created_at"))
    account_age_days = (now - created).days if not pd.isna(created) else 0
    followers = user.get("followers", 0)
    avg_stars = np.mean([r.get("stargazers_count",0) for r in repos]) if repos else 0
    avg_forks = np.mean([r.get("forks_count",0) for r in repos]) if repos else 0
    repo_count = len(repos)
    pushes = [pd.to_datetime(r.get("pushed_at")) for r in repos if r.get("pushed_at")]
    days_since_recent_push = (now - max(pushes)).days if pushes else 3650

    recency_decay = np.exp(-days_since_recent_push / 90.0)
    pop_followers = np.log1p(followers)
    pop_stars = np.log1p(avg_stars)

    popularity_score = ((pop_followers) + (pop_stars)) * 50.0
    signal_score = (recency_decay + (repo_count / 100.0) + (avg_forks / 10.0)) * 33.3333

    features = pd.DataFrame([{
        "followers": followers,
        "avg_stars": avg_stars,
        "avg_forks": avg_forks,
        "repo_count": repo_count,
        "days_since_recent_push": days_since_recent_push,
        "account_age_days": account_age_days,
        "popularity_score": popularity_score,
        "signal_score": signal_score
    }])
    return features, {
        "followers": followers,
        "repo_count": repo_count,
        "avg_stars": avg_stars,
        "avg_forks": avg_forks,
        "days_since_recent_push": days_since_recent_push,
        "recency_decay": recency_decay
    }

st.sidebar.header("Options")
username = st.sidebar.text_input("GitHub username", "torvalds")
use_token = st.sidebar.checkbox("Use GITHUB_TOKEN from env", value=bool(os.getenv("GITHUB_TOKEN")))
analyze_btn = st.sidebar.button("Analyze profile")

if analyze_btn:
    st.info(f"Fetching data for `{username}` ...")
    user = get_user(username)
    if not user:
        st.error("User not found or API rate-limited. Check username and token.")
    else:
        repos = get_repos(username)
        features_df, raw = extract_features(user, repos)

        pred = model.predict(features_df)[0]
        pred_label = label_enc.inverse_transform([int(pred)])[0]
        proba = model.predict_proba(features_df)[0]

        c1, c2 = st.columns([1,3])
        with c1:
            st.image(user.get("avatar_url",""), width=110)
        with c2:
            st.header(f"{user.get('login')} — {pred_label}")
            st.write(user.get("bio",""))
            st.write(f"Followers: {raw['followers']} — Public repos: {raw['repo_count']}")

        st.metric("Predicted activity", pred_label)
        probs_df = pd.DataFrame({"label": label_enc.classes_, "probability": np.round(proba,3)})
        st.table(probs_df.sort_values("probability", ascending=False).reset_index(drop=True))

        st.subheader("Score breakdown (approx.)")
        st.write(f"Popularity score (approx): {features_df['popularity_score'].iloc[0]:.1f}")
        st.write(f"Signal score (approx): {features_df['signal_score'].iloc[0]:.1f}")
        st.write(f"Days since last push: {raw['days_since_recent_push']} (recency decay {raw['recency_decay']:.3f})")

        note = (f"{username}: {pred_label} — {raw['repo_count']} repos, "
                f"{raw['followers']} followers, last push {raw['days_since_recent_push']} days ago.")
        st.text_area("Recruiter note", note, height=90)
        st.download_button("Download note (txt)", note, file_name=f"{username}_note.txt")

        # SHAP / fallback explainability
        show_shap = st.sidebar.checkbox("Show SHAP explainability", value=False)
        if show_shap:
            st.subheader("🔍 Model Explainability (SHAP / fallback)")
            try:
                import shap
                @st.cache_resource
                def _get_explainer(m):
                    try:
                        return shap.TreeExplainer(m)
                    except Exception:
                        return shap.Explainer(m)
                explainer = _get_explainer(model)
                shap_values = explainer(features_df)
                fig, ax = plt.subplots(figsize=(6,3))
                shap.plots.bar(shap_values[0], show=False, max_display=8)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning("SHAP unavailable — showing fallback approximation.")
                try:
                    # get feature names & importances if possible
                    feat_names = list(features_df.columns)
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        importances = np.abs(model.coef_).ravel()
                    else:
                        importances = np.ones(len(feat_names))
                    imp = np.array(importances, dtype=float)
                    imp = imp / (imp.sum() + 1e-9)

                    vals = features_df.iloc[0].values.astype(float)
                    norm_vals = (vals - np.min(vals)) / (np.ptp(vals) + 1e-9)
                    contrib = imp * norm_vals
                    contrib_df = pd.DataFrame({"feature": feat_names, "contrib": contrib})
                    contrib_df = contrib_df.sort_values("contrib", ascending=False).head(8)

                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.barh(contrib_df["feature"][::-1], contrib_df["contrib"][::-1])
                    ax.set_xlabel("Approx. contribution (unitless)")
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Explainability fallback failed: {e}")

