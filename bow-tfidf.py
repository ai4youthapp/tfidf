import streamlit as st
import pandas as pd
import string

# --- Page Setup ---
st.set_page_config(page_title="Bag of Words & TF-IDF Visualizer", layout="wide")

st.title("Bag of Words and TF-IDF Visualizer")

# --- Session State for Step Control ---
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

# --- Input Section ---
st.subheader("Enter Three Documents. Type the documents without a punctuation mark at the end.")
doc1 = st.text_area("Document 1",height=60)
doc2 = st.text_area("Document 2",height=60)
doc3 = st.text_area("Document 3",height=60)

# --- Start Button ---
if st.button("Generate Tables"):
    if not doc1 or not doc2 or not doc3:
        st.warning("Please fill in all three documents.")
    else:
        st.session_state.step = 1

# --- Process only if started ---
if st.session_state.step > 0:

    docs = [doc.lower().strip().split() for doc in [doc1, doc2, doc3]]
    vocab = sorted(set(
        word.strip(string.punctuation)
        for doc in docs
        for word in doc
        if word.strip(string.punctuation)
    ))
    num_docs = len(docs)

    # Precompute BoW
    bow = []
    for doc in docs:
        freq = [doc.count(word) for word in vocab]
        bow.append(freq)

    bow_df = pd.DataFrame(
        bow,
        columns=vocab,
        index=["Document 1", "Document 2", "Document 3"]
    )

    df_counts = [sum(1 for doc in docs if word in doc) for word in vocab]
    df_table = pd.DataFrame([df_counts], columns=vocab, index=["Document Frequency"])

    idf_table = pd.DataFrame(
        [{word: f"{num_docs}/{df_counts[i]}" for i, word in enumerate(vocab)}],
        columns=vocab,
        index=["IDF Formula"]
    )

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
        st.markdown(
            "<h2 style='color:#4CAF50;'>🟢 Step 1: Tokenization</h2>",
    unsafe_allow_html=True)
        for i, tokens in enumerate(docs, start=1):
            st.write(f"Document {i}: {tokens}")

        if st.button("Next ➡️ Step 2"):
            st.session_state.step = 2

    # --- STEP 2 ---
    if st.session_state.step >= 2:
        st.markdown(
            " <h2 style='color:#2986cc;'>🔵 Step 2: Vocabulary</h2>",
    unsafe_allow_html=True)
        st.write(" ,      ".join(vocab))

        if st.button("Next ➡️ Step 3"):
            st.session_state.step = 3

    # --- STEP 3 ---
    if st.session_state.step >= 3:
        st.markdown(
            " <h2 style='color:#c90076;'>🔴 Step 3: Bag of Words</h2>",
    unsafe_allow_html=True)
        show_left_aligned(bow_df)

        if st.button("Next ➡️ Step 4"):
            st.session_state.step = 4

    # --- STEP 4 ---
    if st.session_state.step >= 4:
        st.markdown(
            "<h2 style='color:#4CAF50;'> 🟢 Step 4: Document Frequency </h2>",
    unsafe_allow_html=True)
        show_left_aligned(df_table)

        if st.button("Next ➡️ Step 5"):
            st.session_state.step = 5

    # --- STEP 5 ---
    if st.session_state.step >= 5:
        st.markdown(
            "<h2 style='color:#2986cc;'>🔵 Step 5: IDF </h2>",
    unsafe_allow_html=True)
        show_left_aligned(idf_table)

        if st.button("Next ➡️ Step 6"):
            st.session_state.step = 6

    # --- STEP 6 ---
    if st.session_state.step >= 6:
        st.markdown(
            "<h2 style='color:#c90076;'>🔴 Step 6: TF-IDF </h2>",
    unsafe_allow_html=True)
        show_left_aligned(tfidf_df)

        st.success("Tables generated successfully!")
