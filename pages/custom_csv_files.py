from pathlib import Path

import streamlit as st
import torch
from sklearn import preprocessing
import pandas as pd
from transformers import AutoTokenizer

from tokens_extractor import extract_tokens
from model import Model


@st.cache_resource
def load_labels_encoder():
    labels_encoder = preprocessing.LabelEncoder()
    labels_encoder.fit(allowed_labels)
    return labels_encoder


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Elluran/rudoduo")
    model = Model.from_pretrained(
        "Elluran/rudoduo",
        labels_number=len(allowed_labels),
        pretrained_model_name="cointegrated/rubert-tiny2",
        tokenizer=tokenizer,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_data
def load_paths():
    paths = list(sorted(Path("./examples").glob("*.csv")))
    return list(map(str, paths))


st.set_page_config(layout="wide", page_title="test")
allowed_labels = pd.read_csv("labels.csv")["0"].to_list()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model()
labels_encoder = load_labels_encoder()
paths = load_paths()
sep = ","

"""
# Rudoduo demo
"""

csv_file = st.file_uploader("Upload .csv file")

with st.sidebar:
    sep = st.text_input("separator", value=",")


if csv_file:
    df = pd.read_csv(csv_file, sep=sep)

    tokens = extract_tokens(
        df,
        tokenizer,
        num_of_labels=len(df.columns),
        max_tokens=200,
        max_columns=20,
        max_tokens_per_column=200,
    )

    preds = model.predict(torch.tensor([tokens]).to(DEVICE))
    labels = labels_encoder.inverse_transform(preds)

    st.markdown("### Таблица:")
    st.dataframe(df)

    table = r"""  
Оригинальный лейбл | предсказанная метка
---|--- 
"""

    for orig, guess in zip(df.columns, labels):
        table += orig + "|" + guess + "\n"

    st.markdown(table)
