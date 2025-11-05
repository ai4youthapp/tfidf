import streamlit as st
import pandas as pd
import math

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Bag of Words & TF-IDF Visualizer", layout="wide")

st.title("Bag of Words and TF-IDF Visualizer")

# --- Helper to left-align dataframe ---
def show_left_aligned(df):
    """Display a dataframe with all columns center aligned"""
    st.markdown(
        df.style.set_table_styles(
            [{"selector": "th, td", "props": [("text-align", "center")]}]
        ).to_html(),
        unsafe_allow_html=True
    )

# --- Input Section ---
st.subheader("Enter Three Documents.")
doc1 = st.text_area("Document 1")
doc2 = st.text_area("Document 2")
doc3 = st.text_area("Document 3")

# --- Process Button ---
if st.button("Generate Tables"):
    if not doc1 or not doc2 or not doc3:
        st.warning("Please fill in all three documents.")
    else:
        def clean_text(text):
            # Remove punctuation (including full stops) and convert to lowercase
            translator = str.maketrans(",", string.punctuation)
            return text.lower().translate(translator).split()

        docs = [clean_text(doc) for doc in [doc1, doc2, doc3]]


        # --- Step 2: Vocabulary ---
        import string
        vocab = sorted(set(word.strip(string.punctuation) for doc in docs for word in doc if word.strip(string.punctuation)))
        num_docs = len(docs)

        # --- Step 3: Bag of Words (raw frequency) ---
        bow = []
        for doc in docs:
            freq = [doc.count(word) for word in vocab]
            bow.append(freq)

        # --- Step 4: Term Frequency (TF = count / total words) ---
        tf = []
        for doc in bow:
            total_words = sum(doc)
            tf_doc = [round(f / total_words, 3) if total_words else 0 for f in doc]
            tf.append(tf_doc)

        # --- Tokens ---
        st.subheader("Tokens in Each Document")
        for i, tokens in enumerate(docs, start=1):
            st.write(f"**Document {i}:** {tokens}")

        # --- Vocabulary ---
        st.subheader("Vocabulary of Unique Words")
        st.write(", ".join(vocab))

        # --- Combined Bag of Words & Term Frequency Table ---
        st.subheader("Bag of Words and Term Frequency Table (Words as Columns)")

        # Create a DataFrame where each row corresponds to a document
        bow_df = pd.DataFrame(bow, columns=vocab, index=["Document 1", "Document 2", "Document 3"])

        st.markdown("**Document Vector Table/ Term Frequency Table:**")
        show_left_aligned(bow_df)

               
        # --- Step 5: Document Frequency (DF) ---
        st.subheader("Document Frequency (DF) Table")
        df_counts = []
        for word in vocab:
            df_counts.append(sum(1 for doc in docs if word in doc))
        df_table =  pd.DataFrame([df_counts], columns=vocab, index=["Document Frequency"])
        show_left_aligned(df_table)

        # --- Step 6: Inverse Document Frequency (IDF as ratio) ---
        st.subheader("Inverse Document Frequency (IDF) Table")
        idf_ratio = {word: f"{num_docs}/{df_counts[i]}" for i, word in enumerate(vocab)}
        idf_table = pd.DataFrame([idf_ratio], columns=vocab, index=["IDF Formula (No Calculation)"])
        show_left_aligned(idf_table)

        # --- Step 7: TF-IDF Table (Display as TF x log(N/df)) ---
        st.subheader("TF-IDF Table (TF x log(N/df) Format)")
        tfidf_display = []

        for i in range(num_docs):
            tfidf_doc = []
            for j, word in enumerate(vocab):
                tf_count = bow[i][j]
                df_value = df_counts[j]
                if df_value != 0:
                    expression = f"{tf_count} x log({num_docs}/{df_value})"
                else:
                    expression = "0"
                tfidf_doc.append(expression)
            tfidf_display.append(tfidf_doc)

        tfidf_df = pd.DataFrame(tfidf_display, columns=vocab,
                                index=[f"Doc{i+1}" for i in range(num_docs)])
        show_left_aligned(tfidf_df)

        st.success("Tables generated successfully!")
