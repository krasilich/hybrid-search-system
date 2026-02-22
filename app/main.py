import os
import math
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from openai import OpenAI

# ==========================================
# Configuration & Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="E-commerce Hybrid Search API",
    description="API for Master's Thesis: Hybrid Search + Business-Aware Reranking",
    version="1.0.0"
)

ES_URL = os.getenv("ELASTIC_URL", "http://es01:9200")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

es = Elasticsearch(ES_URL)
client = OpenAI(
    api_key=OPENAI_KEY,
    timeout=10.0,
    max_retries=1
)

INDEX_NAME = "products_hybrid_v1"


# ==========================================
# Pydantic Models (Request / Response Schemas)
# ==========================================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    alpha: float = 0.5  # 0.0 = Vector only, 1.0 = Lexical only, 0.5 = Balanced
    apply_business_reranking: bool = True
    apply_qam: bool = True


class DebugInfo(BaseModel):
    hybrid_score: float  # The raw IR score (BM25 + kNN)
    business_boost: float  # The calculated multiplier (Margin + CVR)
    final_score: float  # hybrid_score * business_boost


class ProductResponse(BaseModel):
    uniq_id: str
    title_en: str
    title_ua: str
    price: float
    margin: float
    conversion_rate: float
    in_stock: bool
    debug_info: DebugInfo


class SearchResponse(BaseModel):
    total_hits: int
    results: List[ProductResponse]
    latency_ms: int


# ==========================================
# Helper Functions
# ==========================================
def get_query_embedding(text: str) -> List[float]:
    """Generates a dense vector for the search query using OpenAI."""
    try:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI Embedding Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")


# ==========================================
# QAM
# ==========================================
KNOWN_BRANDS = ["lego", "barbie", "hasbro", "hot wheels", "disney", "playmobil", "funko", "marvel"]


def perform_qam(query: str) -> dict:
    """
    Performs Query Attribute Mapping (QAM) to extract structured information from the search query.
    For simplicity, this function uses a rule-based approach to identify known brands in the query.
    In a production system, this could be replaced with a more sophisticated NLP model or an LLM-based approach.
    """
    q_lower = query.lower()
    extracted_brand = None

    for brand in KNOWN_BRANDS:
        if brand in q_lower:
            extracted_brand = brand
            break

    return {
        "original_query": query,
        "extracted_brand": extracted_brand
    }


# ==========================================
# Main Search Endpoint
# ==========================================
@app.post("/search", response_model=SearchResponse)
def perform_search(req: SearchRequest):
    logger.info(f"Processing search query: '{req.query}'")

    qam_result = None
    if req.apply_qam:
        qam_result = perform_qam(req.query)
        logger.info(f"QAM Result: {qam_result}")
    else:
        logger.info("QAM disabled for this request.")

    query_vector = get_query_embedding(req.query)

    lexical_weight = req.alpha
    vector_weight = 1.0 - req.alpha

    bool_query = {
        "should": [
            {
                "multi_match": {
                    "query": req.query,
                    "fields": ["title_ua^2", "title_en", "description_ua"],
                    "boost": lexical_weight
                }
            },
            {
                "knn": {
                    "field": "title_vector",
                    "query_vector": query_vector,
                    "num_candidates": 50,
                    "boost": vector_weight
                }
            }
        ],
        "minimum_should_match": 1,
        "filter": []
    }

    if qam_result and qam_result.get("extracted_brand"):
        brand = qam_result["extracted_brand"]
        brand_filter = {
            "nested": {
                "path": "attributes",
                "query": {
                    "bool": {
                        "must": [
                            # Точний пошук по ключу "brand"
                            { "term": { "attributes.key": "brand" } },
                            # Регістронезалежний пошук по частині слова (на випадок "Hasbro Gaming")
                            { "wildcard": {
                                "attributes.value": {
                                    "value": f"*{brand}*",
                                    "case_insensitive": True
                                }
                            }}
                        ]
                    }
                }
            }
        }
        bool_query["filter"].append(brand_filter)

    hybrid_query = {"bool": bool_query}

    if req.apply_business_reranking:
        painless_script = """
        double base_score = _score;
        
        double margin = doc['business_features.margin'].value;
        double cvr = doc['business_features.conversion_rate'].value;
        boolean in_stock = doc['business_features.in_stock'].value;
        
        // Logarithmic smoothing for margin
        double margin_boost = Math.log(1.0 + (margin * 10.0));
        
        // Sigmoid-like business boosting
        double biz_boost = 1.0 + (0.5 * margin_boost) + (0.3 * cvr);
        
        // Hard penalty for out-of-stock items
        double stock_penalty = in_stock ? 1.0 : 0.01;
        
        return base_score * biz_boost * stock_penalty;
        """

        final_query = {
            "function_score": {
                "query": hybrid_query,
                "script_score": {
                    "script": {
                        "source": painless_script
                    }
                },
                "boost_mode": "replace"
            }
        }
    else:
        final_query = hybrid_query

    try:
        es_response = es.search(
            index=INDEX_NAME,
            query=final_query,
            size=req.top_k,
            _source=["uniq_id", "title_en", "title_ua", "business_features"]
        )
    except Exception as e:
        logger.error(f"Elasticsearch Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    hits = es_response["hits"]["hits"]
    total_hits = es_response["hits"]["total"]["value"]
    took_ms = es_response["took"]

    results = []
    for hit in hits:
        source = hit["_source"]
        biz = source.get("business_features", {})

        margin = biz.get("margin", 0.0)
        cvr = biz.get("conversion_rate", 0.0)
        in_stock = biz.get("in_stock", False)
        final_score = hit["_score"]

        if req.apply_business_reranking:
            margin_boost_calc = math.log(1.0 + (margin * 10.0))
            biz_boost_calc = 1.0 + (0.5 * margin_boost_calc) + (0.3 * cvr)
            stock_penalty_calc = 1.0 if in_stock else 0.01

            total_boost = biz_boost_calc * stock_penalty_calc
            hybrid_score = final_score / total_boost if total_boost > 0 else 0.0
        else:
            total_boost = 1.0
            hybrid_score = final_score

        debug_info = DebugInfo(
            hybrid_score=round(hybrid_score, 4),
            business_boost=round(total_boost, 4),
            final_score=round(final_score, 4)
        )

        results.append(ProductResponse(
            uniq_id=source.get("uniq_id", ""),
            title_en=source.get("title_en", ""),
            title_ua=source.get("title_ua", ""),
            price=biz.get("price", 0.0),
            margin=margin,
            conversion_rate=cvr,
            in_stock=in_stock,
            debug_info=debug_info
        ))

    return SearchResponse(
        total_hits=total_hits,
        results=results,
        latency_ms=took_ms
    )


@app.get("/health")
def health_check():
    """System status check."""
    es_status = "ok" if es.ping() else "error"
    return {"status": "running", "elasticsearch": es_status}
