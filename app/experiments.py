import requests
import pandas as pd

API_URL = "http://localhost:8000/search"

def run_search(query: str, alpha: float, apply_biz: bool, apply_qam: bool = True, top_k: int = 5):
    payload = {
        "query": query, "top_k": top_k, "alpha": alpha,
        "apply_business_reranking": apply_biz, "apply_qam": apply_qam
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()["results"]
    except Exception as e:
        print(f"Помилка API: {e}")
        return []

def calc_expected_revenue(results, k):
    rev = 0.0
    for r in results[:k]:
        if r["in_stock"]:
            rev += r["price"] * r["margin"] * r["conversion_rate"]
    return rev

def calc_precision_at_k(results, target_brand, k):
    if not results: return 0.0
    relevant_count = 0
    for r in results[:k]:
        title = r["title_en"].lower()
        # Додаємо відомі саб-бренди Hasbro для чесної перевірки
        if target_brand.lower() in title or "yahtzee" in title or "cranium" in title:
            relevant_count += 1
    return relevant_count / min(len(results), k)

def calc_cl_mrr(results, relevant_en_keywords):
    for i, r in enumerate(results):
        title_lower = r["title_en"].lower()
        if any(kw in title_lower for kw in relevant_en_keywords):
            return 1.0 / (i + 1)
    return 0.0

def experiment_1_qam_effect():
    print("\n" + "="*60)
    print("ЕКСПЕРИМЕНТ 1: Вплив QAM (Оцінка Precision@15)")
    print("="*60)

    test_query = "настільна гра hasbro"
    k = 15 # ЗБІЛЬШИЛИ ДО 15
    res_no_qam = run_search(test_query, alpha=0.5, apply_biz=False, apply_qam=False, top_k=k)
    res_with_qam = run_search(test_query, alpha=0.5, apply_biz=False, apply_qam=True, top_k=k)

    p_no_qam = calc_precision_at_k(res_no_qam, "hasbro", k)
    p_with_qam = calc_precision_at_k(res_with_qam, "hasbro", k)

    print(f"\nБЕЗ QAM (Precision@{k} = {p_no_qam:.2f}):")
    # Виводимо тільки проблемні місця, щоб не спамити консоль
    print(f"  Знайдено товарів: {len(res_no_qam)}")

    print(f"\nЗ QAM (Precision@{k} = {p_with_qam:.2f}):")
    print(f"  Знайдено товарів: {len(res_with_qam)}")

def experiment_2_cross_lingual():
    print("\n" + "="*60)
    print("ЕКСПЕРИМЕНТ 2: Cross-lingual Search (Оцінка CL-MRR)")
    print("="*60)
    test_query = "іграшковий космічний корабель"
    relevant_keywords = ["space", "trek", "star", "spaceship", "rocket", "shuttle", "falcon", "equinox", "mothership"]
    k = 3

    modes = {"Тільки BM25 (alpha=1.0)": 1.0, "Тільки Вектор (alpha=0.0)": 0.0, "Гібрид (Відкалібрований, alpha=0.03)": 0.03}

    for name, alpha in modes.items():
        results = run_search(test_query, alpha=alpha, apply_biz=False, apply_qam=False, top_k=k)
        mrr = calc_cl_mrr(results, relevant_keywords)
        print(f"\n{name} | CL-MRR: {mrr:.2f}")
        for i, r in enumerate(results[:k]):
            print(f"  {i+1}. {r['title_en'][:50]} (Score: {r['debug_info']['final_score']})")

def experiment_3_business_impact():
    print("\n" + "="*60)
    print("ЕКСПЕРИМЕНТ 3: Вплив Business Reranking (Оцінка Rev@5)")
    print("="*60)

    test_query = "barbie"
    fetch_k = 20
    eval_k = 5

    res_base = run_search(test_query, alpha=0.5, apply_biz=False, top_k=fetch_k)
    res_biz = run_search(test_query, alpha=0.5, apply_biz=True, top_k=fetch_k)

    rev_base = calc_expected_revenue(res_base, eval_k)
    rev_biz = calc_expected_revenue(res_biz, eval_k)

    print(f"\n--- БЕЗ Бізнес-бустингу (ТОП-5 з 20) (Rev@{eval_k} = ${rev_base:.2f}) ---")
    df_base = pd.DataFrame([{"Title": r["title_en"][:35], "Price": f"${r['price']:.2f}", "Margin": f"{r['margin']*100:.0f}%", "CVR": f"{r['conversion_rate']*100:.1f}%", "Exp.Rev": f"${(r['price'] * r['margin'] * r['conversion_rate']):.2f}"} for r in res_base[:eval_k]])
    print(df_base.to_string(index=False))

    print(f"\n--- З Бізнес-бустингом (ТОП-5 з 20) (Rev@{eval_k} = ${rev_biz:.2f}) ---")
    df_biz = pd.DataFrame([{"Title": r["title_en"][:35], "Price": f"${r['price']:.2f}", "Margin": f"{r['margin']*100:.0f}%", "CVR": f"{r['conversion_rate']*100:.1f}%", "Exp.Rev": f"${(r['price'] * r['margin'] * r['conversion_rate']):.2f}"} for r in res_biz[:eval_k]])
    print(df_biz.to_string(index=False))

    diff = rev_biz - rev_base
    pct = (diff / rev_base * 100) if rev_base > 0 else 0
    print(f"\nРізниця очікуваного доходу (Rev@{eval_k}): +${diff:.2f} (+{pct:.1f}%)")

if __name__ == "__main__":
    experiment_1_qam_effect()
    experiment_2_cross_lingual()
    experiment_3_business_impact()