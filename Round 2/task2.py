import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertModel
from PyPDF2 import PdfReader
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
conference_keywords = {
    "CVPR": [
        "vision", "image", "object detection", "computer vision", "segmentation", "cnn", 
        "visual recognition", "image classification", "scene understanding", "semantic segmentation",
        "face recognition", "pose estimation", "generative models", "image synthesis", 
        "augmented reality", "3D vision", "video analysis", "motion detection", "neural networks"
    ],
    "EMNLP": [
        "NLP", "language", "text generation", "translation", "bert", "syntax", "semantics", 
        "language modeling", "sentiment analysis", "question answering", "text classification", 
        "named entity recognition", "speech recognition", "dialog systems", "language representation", 
        "neural machine translation", "word embeddings", "transformers", "pretrained models"
    ],
    "KDD": [
        "data mining", "big data", "clustering", "analytics", "patterns", "data science", 
        "machine learning", "data visualization", "anomaly detection", "graph mining", "data privacy", 
        "recommender systems", "social network analysis", "predictive modeling", "classification", 
        "regression", "data preprocessing", "feature selection", "model evaluation"
    ],
    "NeurIPS": [
        "deep learning", "neural networks", "reinforcement learning", "generative", "ai", "optimization", 
        "supervised learning", "unsupervised learning", "transfer learning", "multi-agent systems", 
        "meta-learning", "fairness", "interpretability", "scalability", "computational neuroscience", 
        "learning theory", "evolutionary algorithms", "vision", "robotics", "AI ethics"
    ],
    "TMLR": [
        "theory", "machine learning", "algorithms", "mathematical analysis", "theoretical models", 
        "optimization", "generalization", "bias-variance tradeoff", "PAC learning", "statistical learning", 
        "information theory", "stochastic processes", "convex optimization", "online learning", 
        "sample complexity", "game theory", "model selection", "learning dynamics", "gradient methods"
    ]
}

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def extract_text_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = bert_model(**tokens)
    embeddings = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings


def generate_rationale(text, conference_keywords):
    rationale = []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    word_scores = tfidf_matrix.toarray().flatten()
    top_keywords = [feature_names[i] for i in word_scores.argsort()[-5:]] 
    for conf, keywords in conference_keywords.items():
        matched_keywords = [kw for kw in top_keywords if kw.lower() in keywords]
        if matched_keywords:
            rationale.append(f"Contains key terms for {conf}: {', '.join(matched_keywords)}")
    if not rationale:
        rationale.append("General relevance to the topic")
    
    return ", ".join(rationale) if rationale else "NA"

def summarize_abstract(text):
    sentences = text.split(".")
    summary = " ".join(sentences[:1]) 
    return summary.strip() if summary else "Summary not available"


def train_model(csv_path, pdf_folder, model_path="rf_model_with_rationale.pkl"):
    data = pd.read_csv(csv_path)
    train_data = data[data["Publishable"] == 1]
    embeddings = []
    conferences = []
    for _, row in train_data.iterrows():
        pdf_path = os.path.join(pdf_folder, row["Paper Id"])
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue 
        embedding = extract_text_embeddings(text)
        embeddings.append(embedding)
        conferences.append(row["Conference"] if row["Conference"] != "NA" else "NA")
    X_train = np.array(embeddings)
    y_train = np.array(conferences)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")
    
    return rf_model

def run_inference(csv_path, pdf_folder, model_path="rf_model_with_rationale.pkl"):
    if os.path.exists(model_path):
        rf_model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Trained model not found, please run the training loop first.")
        return
    papers_df = pd.read_csv(csv_path)
    publishable_papers = papers_df[papers_df["Publishable"] == 1]
    non_publishable_papers = papers_df[papers_df["Publishable"] == 0]
    output = []
    pdf_files = [file for file in os.listdir(pdf_folder) if file.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        paper_info = papers_df[papers_df["Paper Id"] == pdf_file]
        if paper_info.empty:
            continue  
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue 
        if not paper_info.empty and paper_info.iloc[0]["Publishable"] == 1:
            embedding = extract_text_embeddings(text)
            conference_pred = rf_model.predict([embedding])[0]
            rationale = generate_rationale(text, conference_keywords) if conference_pred != "NA" else "NA"
            summary = summarize_abstract(text)
        else:
            conference_pred = "NA"
            rationale = "NA"
            summary = "NA"
        
        output.append([pdf_file, paper_info.iloc[0]["Publishable"], conference_pred, rationale, summary])
    
    non_pdf_papers = non_publishable_papers[~non_publishable_papers["Paper Id"].isin(pdf_files)]
    for _, row in non_pdf_papers.iterrows():
        output.append([row["Paper Id"], 0, "NA", "NA", "NA"])
    
    results_df = pd.DataFrame(output, columns=["Paper Id", "Publishable", "Conference", "Rationale", "Summary"])
    results_df.to_csv("result.csv", index=False)
    print("Predictions saved to result.csv")



# Main 
if __name__ == "__main__":
    pdf_folder = "Papers"  # Path to folder containing PDF files
    csv_path = "result1.csv"  # Path to training CSV file  change with csv containing paper id and publishable
    
    # uncomment to train the model 
    # train_model(csv_path, pdf_folder)
    
    # Run inference with the trained model
    run_inference(csv_path,pdf_folder)
