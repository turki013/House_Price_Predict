import os
import sys
import re
import json
import joblib
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\HUAWEI\House_price"
sys.path.append(BASE_DIR)

from aqar_benchmark_mapper import AqarBenchmarkMapper

# 1. Load pipeline and mapper
pipeline_bundle = joblib.load(os.path.join(BASE_DIR, "house_price_pipeline.joblib"))
encoder = pipeline_bundle["target_encoder"]
model = pipeline_bundle["model"]
feature_names = pipeline_bundle["feature_names"]
benchmark_mapper = AqarBenchmarkMapper(os.path.join(BASE_DIR, "aqar_benchmarks_2026.json"))

# 2. Load frozen 45 live listings
with open(os.path.join(BASE_DIR, "live_eval_dataset.json"), "r", encoding="utf-8") as f:
    listings = json.load(f)

print(f"Loaded {len(listings)} frozen live listings for evaluation.")

arabic_digits = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9'}
def normalize_text(s):
    for k, v in arabic_digits.items():
        s = s.replace(k, v)
    return s

def extract_features_from_text(item):
    full_text = normalize_text(item.get("full_text", "") + " " + item.get("title", ""))
    
    def extract_master(t):
        m = re.search(r'(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)', t)
        if m:
            c = int(m.group(1))
            if 1 <= c <= 8: return float(c)
        if 'ماستر' in t or 'جناح رئيسي' in t: return 1.0
        return 0.0

    def extract_living(t):
        count = 0
        m_majlis = re.search(r'(\d+)\s*مجالس', t)
        if m_majlis: count += int(m_majlis.group(1))
        elif 'مجلسين' in t: count += 2
        elif 'مجلس' in t: count += 1
        m_sala = re.search(r'(\d+)\s*صالات', t)
        if m_sala: count += int(m_sala.group(1))
        elif 'صالتين' in t: count += 2
        elif 'صالة' in t or 'صاله' in t: count += 1
        return float(min(count, 8))

    def extract_facade(t):
        if re.search(r'شارعين|زاوية|زاويه|رأس بلك|راس بلك|ثلاث شوارع', t): return 'corner'
        elif re.search(r'شمالي|شماليه|شمالية', t): return 'north'
        elif re.search(r'شرقي|شرقيه|شرقية', t): return 'east'
        elif re.search(r'جنوبي|جنوبيه|جنوبية', t): return 'south'
        elif re.search(r'غربي|غربيه|غربية', t): return 'west'
        return 'unknown'

    def extract_street_width(t):
        m = re.search(r'(?:شارع|عرض)\s*(\d{2,3})', t)
        if m:
            w = int(m.group(1))
            if 8 <= w <= 100: return float(w)
        return 15.0

    def extract_extra_apts(t):
        if 'شقتين' in t or 'مع شقتين' in t or 'وشقتين' in t: return 2
        elif 'شقة' in t or 'مع شقة' in t or 'وشقة' in t: return 1
        return 0

    facade_val = extract_facade(full_text)

    row = {
        'size': float(item['size']),
        'bedrooms': float(item.get('bedrooms', 4)),
        'bathrooms': float(item.get('bathrooms', 4)),
        'is_villa': 1,
        'is_apartment': 0,
        'has_driver_room': int(bool(re.search(r'سائق|سواق', full_text))),
        'has_maid_room': int(bool(re.search(r'خادمة|شغالة', full_text))),
        'is_new': int(bool(re.search(r'جديد|جديدة|مشروع', full_text))),
        'has_elevator': int(bool(re.search(r'مصعد|اصنصير', full_text))),
        'has_pool': int(bool(re.search(r'مسبح', full_text))),
        'has_annex': int(bool(re.search(r'ملحق', full_text))),
        'has_garage': int(bool(re.search(r'مدخل سيارة|مدخل سياره|كراج', full_text))),
        'has_central_ac': int(bool(re.search(r'تكييف مخفي|تكيف مخفي|مركزي|سبلت مخفي', full_text))),
        'has_duplex_stairs': int(bool(re.search(r'درج صالة|درج صاله|درج داخلي|دوبلكس', full_text))),
        'street_width': extract_street_width(full_text),
        'extra_apartments': extract_extra_apts(full_text),
        'master_bedrooms': extract_master(full_text),
        'living_rooms': extract_living(full_text),
        'has_double_height': int(bool(re.search(r'دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع', full_text))),
        'elevator_installed': int(bool(re.search(r'مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب', full_text))),
        'elevator_ready': int(bool(re.search(r'تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد', full_text))),
        'has_custom_built': int(bool(re.search(r'بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام', full_text))),
        'has_spacious_yard': int(bool(re.search(r'حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه', full_text))),
        'is_modern': int(bool(re.search(r'مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن', full_text))),
        'facade_corner': 1 if facade_val == 'corner' else 0,
        'facade_east': 1 if facade_val == 'east' else 0,
        'facade_north': 1 if facade_val == 'north' else 0,
        'facade_south': 1 if facade_val == 'south' else 0,
        'facade_unknown': 1 if facade_val == 'unknown' else 0,
        'facade_west': 1 if facade_val == 'west' else 0,
        'neighborhood': benchmark_mapper.clean_district(item['district'])
    }
    return row

eval_records = []

for item in listings:
    actual_price = float(item["price"])
    size = float(item["size"])
    dist = item["district"]
    
    # Extract features and predict base historical price
    feat = extract_features_from_text(item)
    df_feat = pd.DataFrame([feat])
    df_feat['neighborhood_encoded'] = encoder.transform(df_feat[['neighborhood']])
    pred_log = model.predict(df_feat[feature_names])
    pred_hist = float(np.expm1(pred_log)[0])
    
    # Get Aqar benchmark information
    bm = benchmark_mapper.get_benchmark(dist, size)
    bm_2026_avg = bm["avg_price_2026"]
    bm_hist_avg = bm["hist_price_base"]
    growth_ratio = bm["growth_ratio"]
    
    # Method 1: Pure Aqar Benchmark Alone
    p_m1 = bm_2026_avg
    
    # Method 2: Model Valuation × District Growth Index
    p_m2 = pred_hist * growth_ratio
    
    # Method 3: Relative Ratio to District Mean (Indexed Benchmark Modulation)
    rel_index = pred_hist / bm_hist_avg if bm_hist_avg > 0 else 1.0
    p_m3 = bm_2026_avg * rel_index
    
    eval_records.append({
        "district": dist,
        "size": size,
        "actual_price": actual_price,
        "pred_hist": pred_hist,
        "bm_2026_avg": bm_2026_avg,
        "bm_hist_avg": bm_hist_avg,
        "growth_ratio": growth_ratio,
        "pred_m1": p_m1,
        "pred_m2": p_m2,
        "pred_m3": p_m3,
        "err_m1": (p_m1 - actual_price) / actual_price,
        "err_m2": (p_m2 - actual_price) / actual_price,
        "err_m3": (p_m3 - actual_price) / actual_price,
    })

df_eval = pd.DataFrame(eval_records)

# Metrics calculation
def calc_metrics(errors):
    bias = np.median(errors) * 100
    mape = np.mean(np.abs(errors)) * 100
    within_15 = np.mean(np.abs(errors) <= 0.15) * 100
    within_25 = np.mean(np.abs(errors) <= 0.25) * 100
    return {
        "Bias (Median Error %)": round(bias, 2),
        "MAPE (%)": round(mape, 2),
        "Within ±15% Accuracy (%)": round(within_15, 1),
        "Within ±25% Accuracy (%)": round(within_25, 1)
    }

metrics_m1 = calc_metrics(df_eval["err_m1"])
metrics_m2 = calc_metrics(df_eval["err_m2"])
metrics_m3 = calc_metrics(df_eval["err_m3"])

summary = [
    {"Method": "1. متوسط عقار لوحده (Aqar Benchmark Alone)", **metrics_m1},
    {"Method": "2. الموديل × النمو (Model Base × District Growth)", **metrics_m2},
    {"Method": "3. النسبة للمتوسط (Indexed Benchmark Modulation)", **metrics_m3}
]

df_summary = pd.DataFrame(summary)

print("\n" + "=" * 90)
print("EMPIRICAL COMPARISON ON 45 LIVE LISTINGS (PRE-DECLARED SELECTION RULE)")
print("=" * 90)
print(df_summary.to_string(index=False))

print("\n" + "=" * 90)
print("SAMPLE LISTINGS BREAKDOWN (FIRST 10 CASES)")
print("=" * 90)
sample_view = df_eval[["district", "size", "actual_price", "pred_m1", "pred_m2", "pred_m3", "err_m1", "err_m2", "err_m3"]].head(10).copy()
for col in ["err_m1", "err_m2", "err_m3"]:
    sample_view[col] = sample_view[col].apply(lambda x: f"{x*100:+.1f}%")
print(sample_view.to_string(index=False))

# Export results
eval_json_path = os.path.join(BASE_DIR, "live_benchmark_evaluation_results.json")
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "selection_rule": {
            "target": "Active residential villa listings in Riyadh from Aqar",
            "districts": ["النرجس", "العارض", "الياسمين"],
            "size_bounds": [150, 650],
            "price_bounds": [1200000, 8500000],
            "sample_size": len(df_eval)
        },
        "metrics_comparison": summary,
        "records": eval_records
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved evaluation results to: {eval_json_path}")
