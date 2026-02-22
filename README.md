# 🔍 E-Commerce Hybrid Search Engine with Business-Aware Reranking

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.12-005571?logo=elasticsearch)](https://www.elastic.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai)](https://openai.com/)

This repository contains the source code for the Master's Thesis project: **"Research of Ranking Algorithms for a Hybrid Product Search System in Multilingual E-commerce Platforms."**

## 📖 About the Project
Modern e-commerce platforms struggle with two main challenges:
1. **The Vocabulary Mismatch Problem** in multilingual environments (e.g., a user searches in Ukrainian, but the database is in English).
2. **The Relevance vs. Revenue Conflict**, where models optimize only for clicks (CTR/NDCG) while ignoring business priorities like profit margins and conversion rates.

This project introduces a **Hybrid Search Architecture** that solves these issues by combining lexical search (BM25), dense vector search (k-NN via OpenAI embeddings), Query Attribute Modeling (QAM), and a custom Business-Aware Reranking algorithm executed directly in the database.

## ✨ Key Features
* **Query Attribute Modeling (QAM):** Extracts entities (like brands) from user queries to apply hard boolean filters, effectively eliminating semantic noise.
* **Cross-Lingual Vector Search:** Projects queries and products into a shared latent space using `text-embedding-3-small`, enabling users to search in their native language without needing to translate the entire database.
* **Dynamic Score Blending:** A linear combination of BM25 and Cosine Similarity scores.
* **Business-Aware Reranking (LTR Alternative):** Uses an Elasticsearch `Painless` script to dynamically boost high-margin and highly convertible products in real-time without sacrificing user relevance.

## 🏗️ Architecture
1. **ETL Pipeline (`loader.py`)**: Augments a raw dataset using LLMs (`gpt-4o-mini`), generates embeddings, synthesizes business metrics (Margin, CVR, Stock), and indexes data into Elasticsearch.
2. **Search API (`main.py`)**: A FastAPI backend that processes user queries, runs QAM, and executes the hybrid `function_score` query.
3. **Evaluation (`experiments.py`)**: An offline evaluation script calculating `Precision@K`, `CL-MRR` (Cross-Lingual MRR), and `Rev@K` (Expected Revenue).

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose
* Python 3.11+
* OpenAI API Key

### 1. Clone the repository & Setup Environment
```bash
git clone [https://github.com/yourusername/ecommerce-hybrid-search.git](https://github.com/yourusername/ecommerce-hybrid-search.git)
cd ecommerce-hybrid-search

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt