# streamlit_app.py
import streamlit as st, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import re, nltk, joblib, plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from io import StringIO

nltk.download('stopwords')
STOP = set(nltk.corpus.stopwords.words('english'))

st.set_page_config(page_title="Resume–Job Match Dashboard", layout="wide")
st.title("Resume–Job Match Score Dashboard")

@st.cache_data
def load_demo():
    return pd.read_csv("resume_job_matching_dataset.csv")

uploaded_file = st.file_uploader("Upload CSV (optional)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_demo()

# ---------- 1. Quick EDA ----------
st.header("📊 1. Quick Glance")
col1, col2 = st.columns(2)
with col1:
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isna().sum().to_dict())
with col2:
    st.write("Score counts:")
    st.write(df['match_score'].value_counts().sort_index())

fig, ax = plt.subplots()
sns.countplot(x='match_score', data=df, palette='viridis', ax=ax)
ax.set_title("Distribution of Match Scores")
st.pyplot(fig)

# ---------- 2. Text Cleaning ----------
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join(w for w in text.split() if w not in STOP)
    return text

df['clean_jd']  = df['job_description'].apply(clean)
df['clean_res'] = df['resume'].apply(clean)
df['text_pair'] = df['clean_jd'] + ' ' + df['clean_res']

# ---------- 3. Model ----------
X_train, X_test, y_train, y_test = train_test_split(
    df['text_pair'], df['match_score'], test_size=.2, random_state=42, stratify=df['match_score'])
vec  = TfidfVectorizer(max_features=20_000, ngram_range=(1,2))
Xtr  = vec.fit_transform(X_train)
Xte  = vec.transform(X_test)
model = Ridge(alpha=1.0)
model.fit(Xtr, y_train)
pred = model.predict(Xte)

mae, r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
st.metric("Baseline MAE", round(mae,3))
st.metric("Baseline R²", round(r2,3))

# ---------- 4. Radar Chart ----------
low  = df[df['match_score'] <= 2]['clean_jd']
high = df[df['match_score'] >= 4]['clean_jd']
skill_list = ['sql','python','tableau','power bi','machine learning','deep learning',
              'nlp','pandas','statistics','docker','git','java','spring','agile','keras','tensorflow']
def skill_freq(series, skills):
    return [series.str.contains(s, regex=False).sum() for s in skills]
low_counts  = skill_freq(low, skill_list)
high_counts = skill_freq(high, skill_list)

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=low_counts,  theta=skill_list, name='Low ≤2', line_color='crimson'))
fig_radar.add_trace(go.Scatterpolar(r=high_counts, theta=skill_list, name='High ≥4', line_color='forestgreen'))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title='Skill Presence Radar')
st.plotly_chart(fig_radar, use_container_width=True)

# ---------- 5. Word Cloud ----------
all_words = ' '.join(high).split()
freq = pd.Series(all_words).value_counts()
wc = WordCloud(width=600, height=300, background_color='black').generate_from_frequencies(freq)
fig_wc, ax = plt.subplots(figsize=(12,6))
ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
ax.set_title('Skills most common in HIGH-score JDs')
st.pyplot(fig_wc)

# ---------- 6. Residuals & True vs Predicted ----------
residuals = y_test - pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
sns.scatterplot(x=pred, y=residuals, ax=ax1)
ax1.axhline(0, ls='--', c='red'); ax1.set_title('Residuals vs Predicted')
sns.scatterplot(x=y_test, y=pred, ax=ax2)
ax2.plot([0,5],[0,5],'--r'); ax2.set_title('True vs Predicted')
st.pyplot(fig)

# ---------- 7. Real-time Prediction ----------
st.header("🔮 Predict Match Score")
jd_input  = st.text_area("Job Description", height=150)
res_input = st.text_area("Resume", height=150)

if st.button("Predict"):
    if jd_input and res_input:
        combined = clean(jd_input) + ' ' + clean(res_input)
        vec_input = vec.transform([combined])
        score = float(model.predict(vec_input)[0])
        st.success(f"Predicted Match Score: **{score:.2f}** / 5")
    else:
        st.warning("Please enter both fields.")