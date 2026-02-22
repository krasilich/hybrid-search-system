import os
import json
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # Прогрес-бар

# ==========================================
# Configuration & Setup
# ==========================================
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

ES_URL = os.getenv("ELASTIC_URL", "http://es01:9200")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

es = Elasticsearch(ES_URL)
client = OpenAI(
    api_key=OPENAI_KEY,
    timeout=15,
    max_retries=1
)

ALIAS_NAME = "products_hybrid_v1"
DATASET_PATH = "amazon_toys.csv"
MAX_WORKERS = 10

# ==========================================
# Helper Functions for Data Cleaning
# ==========================================
def parse_price(price_str):
    try:
        if pd.isna(price_str): return 0.0
        return float(str(price_str).replace('£', '').replace(',', '').strip())
    except:
        return 0.0

def parse_rating(rating_str):
    try:
        if pd.isna(rating_str): return 0.0
        return float(str(rating_str).split()[0])
    except:
        return 0.0

# ==========================================
# Elasticsearch Setup
# ==========================================
def create_index():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_index_name = f"{ALIAS_NAME}_{timestamp}"

    mapping = {
        "settings": { "number_of_shards": 1 },
        "mappings": {
            "properties": {
                "uniq_id": { "type": "keyword" },
                "title_en": { "type": "text", "analyzer": "english" },
                "description_en": { "type": "text", "analyzer": "english" },
                "title_ua": { "type": "text", "analyzer": "ukrainian" },
                "description_ua": { "type": "text", "analyzer": "ukrainian" },
                "title_vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                },
                "business_features": {
                    "properties": {
                        "price": { "type": "float" },
                        "margin": { "type": "float" },
                        "conversion_rate": { "type": "float" },
                        "in_stock": { "type": "boolean" }
                    }
                },
                "attributes": {
                    "type": "nested",
                    "properties": {
                        "key": { "type": "keyword" },
                        "value": { "type": "keyword" }
                    }
                }
            }
        }
    }
    es.indices.create(index=new_index_name, body=mapping)
    return new_index_name

def switch_alias(new_index_name):
    actions = [{"add": {"index": new_index_name, "alias": ALIAS_NAME}}]
    if es.indices.exists_alias(name=ALIAS_NAME):
        old_indices = es.indices.get_alias(name=ALIAS_NAME).keys()
        for old_idx in old_indices:
            actions.append({"remove": {"index": old_idx, "alias": ALIAS_NAME}})
    es.indices.update_aliases(body={"actions": actions})

# ==========================================
# OpenAI API Integration
# ==========================================
def translate_to_ukrainian(title, description):
    prompt = f"""
    Translate title: '{title}'
    Translate description: '{description}'
    
    Return the result ONLY as a JSON object with exactly two keys: "title_ua" and "description_ua".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert e-commerce localization specialist for the Ukrainian market. "
                        "Your task is to translate product titles and descriptions from English to Ukrainian. Rules:\n"
                        "1. Keep brand names (Lego, Barbie, Hot Wheels, Sony) in English.\n"
                        "2. Keep model numbers (e.g., 'RX-500') in English.\n"
                        "3. Adapt measurement units to metric (inches -> cm) if applicable.\n"
                        "4. Use SEO-friendly terminology typical for Rozetka/Prom.ua."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"title_ua": title, "description_ua": str(description)}

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        return [0.0] * 1536

# ==========================================
# Task for a single thread
# ==========================================
def process_single_row(row, target_index, idx):
    try:
        uniq_id = str(row.get('uniq_id', f"GEN_{idx}"))
        title_en = str(row.get('product_name', ''))
        desc_en = str(row.get('product_description', ''))
        brand = str(row.get('manufacturer', 'Unknown'))

        price = parse_price(row.get('price'))
        rating = parse_rating(row.get('average_review_rating'))

        ua_data = translate_to_ukrainian(title_en, desc_en)
        text_for_vector = f"{title_en} {brand} {desc_en}"
        vector = get_embedding(text_for_vector)

        margin = max(0.05, min(0.50, round(np.random.normal(0.20, 0.05), 3)))
        conv_rate = max(0.0, min(1.0, round((rating * 0.01) + np.random.uniform(-0.005, 0.005), 4)))
        in_stock = bool(np.random.choice([True, False], p=[0.8, 0.2]))

        doc = {
            "uniq_id": uniq_id,
            "title_en": title_en,
            "description_en": desc_en,
            "title_ua": ua_data.get('title_ua', title_en),
            "description_ua": ua_data.get('description_ua', ''),
            "title_vector": vector,
            "business_features": {
                "price": price,
                "margin": margin,
                "conversion_rate": conv_rate,
                "in_stock": in_stock
            },
            "attributes": [
                {"key": "brand", "value": brand},
                {"key": "category", "value": "Toys & Games"}
            ]
        }

        es.index(index=target_index, id=uniq_id, body=doc)
        return True
    except Exception as e:
        return False

# ==========================================
# Main ETL Pipeline
# ==========================================
def run_etl():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print("Reading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # df_subset = df.head(100).copy()
    df_subset = df.dropna(subset=['product_name']).copy()
    print(f"Processing {len(df_subset)} records using {MAX_WORKERS} threads...")

    target_index = create_index()
    success_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_row, row, target_index, idx): idx
            for idx, row in df_subset.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="ETL Progress"):
            if future.result():
                success_count += 1

    print(f"\nIndexing completed. Successfully indexed {success_count}/{len(df_subset)} records.")

    if success_count > 0:
        switch_alias(target_index)
        print("Alias switched successfully!")

if __name__ == "__main__":
    run_etl()