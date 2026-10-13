import re
import joblib
import streamlit as st

@st.cache_resource
def load():
    bundle = joblib.load("svm_sentiment_pipeline.joblib")
    return bundle["model"], bundle["vectorizer"]

model, vectorizer = load()

def clean_text(t):
    t = re.sub(r"[^a-z0-9\s]", "", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

st.title("Movie Review Sentiment Analysis (SVM)")
review = st.text_area("Movie Reviews: Best Movie in the world, Nice Good story, wonderful acting, Worst Movie, Don't watch this", height=150,
                      placeholder="Enter a movie review here...")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        X = vectorizer.transform([clean_text(review)])
        pred = int(model.predict(X)[0])
        score = float(model.decision_function(X)[0])
        st.subheader("Positive" if pred == 1 else "Negative")
        st.metric("Decision score (distance from boundary)", f"{score:.3f}")
        st.caption("Positive score means positive sentiment. A larger absolute value means a more confident prediction.")
