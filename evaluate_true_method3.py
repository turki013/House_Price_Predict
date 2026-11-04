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

# 1. Load pipelines and mapper
# Standard Model Pipeline (Method 2)
std_pipeline = joblib.load(os.path.join(BASE_DIR, "house_price_pipeline.joblib"))
std_model = std_pipeline["model"]
std_encoder = std_pipeline["target_encoder"]
std_feature_names = std_pipeline["feature_names"]

# True Relative Ratio Model Pipeline (Method 3)
rel_bundle = joblib.load(os.path.join(BASE_DIR, "house_price_relative_pipeline.joblib"))
rel_pipe = rel_bundle["pipe"]
rel_feature_names = rel_bundle["feature_names"]
facade_cols = rel_bundle["facade_columns"]

# Benchmark Mapper
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

def extract_features(item):
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
    
    feat = extract_features(item)
    df_feat = pd.DataFrame([feat])
    
    # Benchmark numbers
    bm = benchmark_mapper.get_benchmark(dist, size)
    bm_2026_avg = bm["avg_price_2026"]
    bm_hist_avg = bm["hist_price_base"]
    growth_ratio = bm["growth_ratio"]
    
    # --- Method 1: Pure Aqar 2026 Benchmark Alone ---
    p_m1 = bm_2026_avg
    
    # --- Method 2: Standard Absolute Model (predicting log price) × Growth Ratio ---
    df_feat_std = df_feat.copy()
    df_feat_std['neighborhood_encoded'] = std_encoder.transform(df_feat_std[['neighborhood']])
    pred_log_std = std_model.predict(df_feat_std[std_feature_names])
    pred_hist_std = float(np.expm1(pred_log_std)[0])
    p_m2 = pred_hist_std * growth_ratio
    
    # --- Method 3: True Relative Model (trained on log(price / base_hist)) × base_2026 ---
    # Pipe takes DataFrame with "neighborhood" and numeric features
    pred_rel_ratio = float(rel_pipe.predict(df_feat[rel_feature_names])[0])
    relative_multiplier = np.exp(pred_rel_ratio)
    p_m3 = bm_2026_avg * relative_multiplier
    
    eval_records.append({
        "district": dist,
        "size": size,
        "actual_price": actual_price,
        "pred_m1": p_m1,
        "pred_m2": p_m2,
        "pred_m3": p_m3,
        "rel_multiplier": relative_multiplier,
        "err_m1": (p_m1 - actual_price) / actual_price,
        "err_m2": (p_m2 - actual_price) / actual_price,
        "err_m3": (p_m3 - actual_price) / actual_price,
    })

df_eval = pd.DataFrame(eval_records)

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

m1 = calc_metrics(df_eval["err_m1"])
m2 = calc_metrics(df_eval["err_m2"])
m3 = calc_metrics(df_eval["err_m3"])

summary = [
    {"Method": "1. متوسط عقار لوحده (Aqar Benchmark Alone)", **m1},
    {"Method": "2. الموديل القياسي × النمو (Standard Model × Growth)", **m2},
    {"Method": "3. نموذج النسبة الحقيقي × مرساة 2026 (True Relative Model)", **m3}
]

df_summary = pd.DataFrame(summary)

print("\n" + "=" * 95)
print("PAIRED EMPIRICAL COMPARISON ON 45 LIVE LISTINGS (FROZEN SAMPLE)")
print("=" * 95)
print(df_summary.to_string(index=False))

print("\n" + "=" * 95)
print("SAMPLE LISTINGS BREAKDOWN (FIRST 10 CASES)")
print("=" * 95)
sample_view = df_eval[["district", "size", "actual_price", "pred_m1", "pred_m2", "pred_m3", "rel_multiplier", "err_m1", "err_m2", "err_m3"]].head(10).copy()
for col in ["err_m1", "err_m2", "err_m3"]:
    sample_view[col] = sample_view[col].apply(lambda x: f"{x*100:+.1f}%")
sample_view["rel_multiplier"] = sample_view["rel_multiplier"].apply(lambda x: f"{x:.3f}x")
print(sample_view.to_string(index=False))

# Paired Differences: Method 3 vs Method 1
err_diff = np.abs(df_eval["err_m3"]) - np.abs(df_eval["err_m1"])
m3_wins = (err_diff < 0).sum()
m1_wins = (err_diff > 0).sum()
ties = (err_diff == 0).sum()
print("\n--- PAIRED HEAD-TO-HEAD BREAKDOWN (Method 3 vs Method 1) ---")
print(f"Listings where Method 3 had lower absolute error: {m3_wins} / {len(df_eval)} ({m3_wins/len(df_eval)*100:.1f}%)")
print(f"Listings where Method 1 had lower absolute error: {m1_wins} / {len(df_eval)} ({m1_wins/len(df_eval)*100:.1f}%)")
print(f"Ties: {ties}")

# Save detailed results
results_path = os.path.join(BASE_DIR, "live_benchmark_evaluation_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "sample_size": len(df_eval),
        "summary": summary,
        "head_to_head": {
            "method3_wins": int(m3_wins),
            "method1_wins": int(m1_wins),
            "ties": int(ties)
        },
        "records": eval_records
    }, f, indent=2, ensure_ascii=False)
