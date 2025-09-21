# chatbot_insights_pipeline.py
"""
Unified Emotion + Mental Health Signal Detection + Knowledge Enrichment Module
Integrates classification signals (emotion, sentiment, toxicity, risk, disorder, medication)
and enhances chatbot response via context-grounded knowledge (RAG from mental health datasets).
"""

import re
from transformers import pipeline
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ------------------ Classifier Pipelines ------------------

try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
    # disorder_classifier = pipeline("text-classification", model="mental/mental-bert-base-uncased")  # Gated model
    disorder_classifier = None  # Temporarily disabled due to gated model access
except Exception as e:
    print(" Model loading error:", e)
    emotion_classifier = sentiment_classifier = toxicity_classifier = disorder_classifier = None

# ------------------ Medication Matching ------------------

medication_keywords = [
    "lithium", "prozac", "zoloft", "xanax", "valium", "lexapro", "wellbutrin",
    "abilify", "seroquel", "celexa", "paxil", "effexor", "lamictal"
]

def detect_medication_mentions(text: str):
    return [med for med in medication_keywords if med.lower() in text.lower()]

# ------------------ Suicide Risk (Keyword-Based Fallback) ------------------

suicidal_keywords = [
    "kill myself", "suicide", "end it all", "die", "can't go on", "give up", "no way out"
]

def detect_suicide_risk(text: str):
    lowered = text.lower()
    return any(kw in lowered for kw in suicidal_keywords)

# ------------------ Analysis Function ------------------

def analyze_user_input(text: str):
    try:
        emotion = emotion_classifier(text)[0][0]['label'] if emotion_classifier else "N/A"
        sentiment = sentiment_classifier(text)[0]['label'] if sentiment_classifier else "N/A"
        toxicity = toxicity_classifier(text)[0]['label'] if toxicity_classifier else "N/A"
        disorder_prediction = disorder_classifier(text)[0]['label'] if disorder_classifier else "N/A"
        medication_mentions = detect_medication_mentions(text)
        suicide_flag = detect_suicide_risk(text)

        return {
            "emotion": emotion,
            "sentiment": sentiment,
            "toxicity": toxicity,
            "disorder_prediction": disorder_prediction,
            "medications_detected": medication_mentions,
            "suicide_risk": "high" if suicide_flag else "low"
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------ Dataset Loader & Vectorizer ------------------

def load_and_index_datasets():
    datasets_to_load = [
        ("tolu07/Mental_Health_FAQ", "FAQ"),
        ("vivekdugale/llama2_mental_health_dataset_172", "LLaMA2 Dialogues"),
        ("jtatman/mental_health_psychology_curated_alpaca", "Alpaca Instructions"),
        ("sayanroy058/Social_Media_And_Twitter_Mental_Health_Dataset", "Social Media Risk")
    ]

    all_docs = []
    for dataset_name, label in datasets_to_load:
        try:
            ds = load_dataset(dataset_name)
            for row in ds["train"]:
                row_text = "\n".join(str(val) for val in row.values())
                all_docs.append(Document(page_content=row_text, metadata={"source": label}))
        except Exception as e:
            print(f"Failed to load {label}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(all_docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="./mental_health_knowledge")
    vectorstore.persist()
    print(" Knowledge base indexed and stored.")

# ------------------ Main ------------------

if __name__ == "__main__":
    user_input = "I've stopped taking my lithium and I feel hopeless and want to end it all."
    print("\n Running Mental Health Analysis:")
    results = analyze_user_input(user_input)
    print(results)

    print("\n Indexing Knowledge Datasets (first time only)...")
    load_and_index_datasets()