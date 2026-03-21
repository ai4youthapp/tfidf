import streamlit as st
import pandas as pd
import string

# --- Page Setup ---
st.set_page_config(page_title="Bag of Words & TF-IDF Visualizer", layout="wide")

st.title("Bag of Words and TF-IDF Visualizer")

# --- Session State ---
if "step" not in st.session_state:
    st.session_state.step = 0

# --- Helper Function ---
def show_left_aligned(df):
    st.markdown(
        df.style.set_table_styles(
            [{"selector": "th, td", "props": [("text-align", "center")]}]
        ).to_html(),
        unsafe_allow_html=True
    )

# --- 🎒 Animated Bag Function ---
def display_vocab_in_bag(vocab):
    words_html = ""

    for i, word in enumerate(vocab):
        delay = round(i * 0.3, 2)
        words_html += f'<span class="word" style="animation-delay:{delay}s;">{word}</span>'

    html = f"""
<style>
.bag {{
    width: 420px;
    min-height: 240px;
    background: #f4a261;
    border-radius: 20px 20px 40px 40px;
    padding: 20px;
    border: 3px solid #8d5524;
    position: relative;
    margin: auto;
    overflow: hidden;
}}

.handle {{
    position: absolute;
    top: -25px;
    left: 30%;
    width: 40%;
    height: 30px;
    background: #8d5524;
    border-radius: 10px;
}}

.word-container {{
    margin-top: 40px;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}}

.word {{
    margin: 5px;
    padding: 6px 10px;
    background: white;
    border-radius: 6px;
    border: 1px solid #ccc;
    opacity: 0;
    animation: drop 0.6s ease forwards;
}}

@keyframes drop {{
    from {{
        transform: translateY(-120px);
        opacity: 0;
    }}
    to {{
        transform: translateY(0);
        opacity: 1;
    }}
}}
</style>

<div class="bag">
    <div class="handle"></div>
    <div class="word-container">{words_html}</div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)

# --- Input Section ---
st.subheader("Enter Three Documents (no punctuation at the end)")
doc1 = st.text_area("Document 1", height=40)
doc2 = st.text_area("Document 2", height=40)
doc3 = st.text_area("Document 3", height=40)

# --- Start Button ---
if st.button("Generate Tables"):
    if not doc1 or not doc2 or not doc3:
        st.warning("Please fill all documents")
    else:
        st.session_state.step = 1

# --- Process ---
if st.session_state.step > 0:

    docs = [doc.lower().strip().split() for doc in [doc1, doc2, doc3]]

    vocab = sorted(set(
        word.strip(string.punctuation)
        for doc in docs
        for word in doc
        if word.strip(string.punctuation)
    ))

    num_docs = len(docs)

    # --- BoW ---
    bow = []
    for doc in docs:
        freq = [doc.count(word) for word in vocab]
        bow.append(freq)

    bow_df = pd.DataFrame(
        bow,
        columns=vocab,
        index=["Document 1", "Document 2", "Document 3"]
    )

    # --- DF ---
    df_counts = [sum(1 for doc in docs if word in doc) for word in vocab]
    df_table = pd.DataFrame([df_counts], columns=vocab, index=["Document Frequency"])

    # --- IDF ---
    idf_table = pd.DataFrame(
        [{word: f"{num_docs}/{df_counts[i]}" for i, word in enumerate(vocab)}],
        columns=vocab,
        index=["IDF Formula"]
    )

    # --- TF-IDF ---
    tfidf_display = []
    for i in range(num_docs):
        row = []
        for j, word in enumerate(vocab):
            tf_count = bow[i][j]
            df_value = df_counts[j]
            row.append(f"{tf_count} x log({num_docs}/{df_value})" if df_value else "0")
        tfidf_display.append(row)

    tfidf_df = pd.DataFrame(
        tfidf_display,
        columns=vocab,
        index=[f"Doc{i+1}" for i in range(num_docs)]
    )

    # --- STEP 1 ---
    if st.session_state.step >= 1:
        st.markdown("<h2 style='color:#4CAF50;'>🟢 Step 1: Tokenization</h2>", unsafe_allow_html=True)
        for i, tokens in enumerate(docs, start=1):
            st.write(f"Document {i}: {tokens}")

        if st.button("Next ➡️ Step 2"):
            st.session_state.step = 2

    # --- STEP 2 (ANIMATED BAG) ---
    if st.session_state.step >= 2:
        st.markdown("<h2 style='color:#2986cc;'>🔵 Step 2: Vocabulary (Bag of Words)</h2>", unsafe_allow_html=True)

        display_vocab_in_bag(vocab)

        if st.button("Next ➡️ Step 3"):
            st.session_state.step = 3

    # --- STEP 3 ---
    if st.session_state.step >= 3:
        st.markdown("<h2 style='color:#c90076;'>🔴 Step 3: Bag of Words</h2>", unsafe_allow_html=True)
        show_left_aligned(bow_df)

        if st.button("Next ➡️ Step 4"):
            st.session_state.step = 4

    # --- STEP 4 ---
    if st.session_state.step >= 4:
        st.markdown("<h2 style='color:#4CAF50;'>🟢 Step 4: Document Frequency</h2>", unsafe_allow_html=True)
        show_left_aligned(df_table)

        if st.button("Next ➡️ Step 5"):
            st.session_state.step = 5

    # --- STEP 5 ---
    if st.session_state.step >= 5:
        st.markdown("<h2 style='color:#2986cc;'>🔵 Step 5: IDF</h2>", unsafe_allow_html=True)
        show_left_aligned(idf_table)

        if st.button("Next ➡️ Step 6"):
            st.session_state.step = 6

    # --- STEP 6 ---
    if st.session_state.step >= 6:
        st.markdown("<h2 style='color:#c90076;'>🔴 Step 6: TF-IDF</h2>", unsafe_allow_html=True)
        show_left_aligned(tfidf_df)

        st.success("🎉 All steps completed!")
