import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='World Cup 2026 Predictions', layout='wide')
st.title('World Cup 2026 Predictions Dashboard')
DATA_DIR = Path('.')

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()

pred = load_csv(DATA_DIR / 'submission_wc2026_group_stage_poisson_lgbm_optuna.csv')
qual = load_csv(DATA_DIR / 'wc2026_group_qualification_probabilities.csv')
imp = load_csv(DATA_DIR / 'feature_importance_lgbm_optuna.csv')

if pred.empty:
    st.error('Prediction file not found. Run the notebook first.')
    st.stop()

pred['date'] = pd.to_datetime(pred['date'], errors='coerce')
groups = sorted(pred['group'].dropna().unique())
selected_groups = st.sidebar.multiselect('Groups', groups, default=groups)
teams = sorted(set(pred['home_team']).union(pred['away_team']))
team = st.sidebar.selectbox('Team', ['All'] + teams)
filtered = pred[pred['group'].isin(selected_groups)].copy()
if team != 'All':
    filtered = filtered[(filtered['home_team'] == team) | (filtered['away_team'] == team)]

st.subheader('Match probabilities')
cols = [c for c in ['date','group','home_team','away_team','lambda_home','lambda_away','p_home_win','p_draw','p_away_win','pred'] if c in filtered.columns]
st.dataframe(filtered[cols].sort_values(['group','date']), use_container_width=True)

st.subheader('Highest-confidence picks')
conf = filtered.copy()
conf['best_pick_prob'] = conf[['p_home_win','p_draw','p_away_win']].max(axis=1)
conf['best_pick'] = conf[['p_home_win','p_draw','p_away_win']].idxmax(axis=1).map({'p_home_win':'Home win','p_draw':'Draw','p_away_win':'Away win'})
st.dataframe(conf.sort_values('best_pick_prob', ascending=False)[cols + ['best_pick','best_pick_prob']].head(30), use_container_width=True)

if not qual.empty:
    st.subheader('Qualification probabilities')
    st.dataframe(qual.sort_values('qualification_prob', ascending=False), use_container_width=True)
    st.bar_chart(qual.set_index('team')[['qualification_prob']])
if not imp.empty:
    st.subheader('Feature importance')
    st.dataframe(imp.head(50), use_container_width=True)
st.caption('Probabilistic forecasts, not betting advice.')
