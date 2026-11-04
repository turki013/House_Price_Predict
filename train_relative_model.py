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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# 1. Load raw data and extract features
DATA_PATH = os.path.join(BASE_DIR, "Aqar_data.csv")
df_raw = pd.read_csv(DATA_PATH)

df = df_raw.copy()
df['neighborhood'] = df['location'].str.replace('حي', '').str.replace('- الرياض', '').str.strip()
df['listTitle'] = df['listTitle'].fillna('')
df['details'] = df['details'].fillna('')
full_text = df['details'] + ' ' + df['listTitle']

arabic_digits = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9'}
def normalize_text(s):
    for k, v in arabic_digits.items():
        s = s.replace(k, v)
    return s

norm_text = full_text.apply(normalize_text)

df['is_villa'] = (df['listTitle'].str.contains('فيلا', na=False) | df['details'].str.contains('فيلا', na=False)).astype(int)
df['is_apartment'] = (df['listTitle'].str.contains('شقة|شقه', regex=True, na=False) | df['details'].str.contains('شقة|شقه', regex=True, na=False)).astype(int)
df['has_driver_room'] = df['details'].str.contains('سائق|سواق', regex=True, na=False).astype(int)
df['has_maid_room'] = df['details'].str.contains('خادمة|شغالة', regex=True, na=False).astype(int)
df['is_new'] = df['details'].str.contains('جديد|جديدة', regex=True, na=False).astype(int)
df['has_elevator'] = norm_text.str.contains('مصعد|اصنصير', regex=True, na=False).astype(int)
df['has_pool'] = norm_text.str.contains('مسبح', na=False).astype(int)
df['has_annex'] = norm_text.str.contains('ملحق', na=False).astype(int)
df['has_garage'] = norm_text.str.contains('مدخل سيارة|مدخل سياره|كراج', regex=True, na=False).astype(int)
df['has_central_ac'] = norm_text.str.contains('تكييف مخفي|تكيف مخفي|مركزي|سبلت مخفي', regex=True, na=False).astype(int)
df['has_duplex_stairs'] = norm_text.str.contains('درج صالة|درج صاله|درج داخلي|دوبلكس', regex=True, na=False).astype(int)

def extract_street_width(text):
    m = re.search(r'(?:شارع|عرض)\s*(\d{2,3})', str(text))
    if m:
        w = int(m.group(1))
        if 8 <= w <= 100: return float(w)
    return 15.0
df['street_width'] = norm_text.apply(extract_street_width)

def extract_extra_apartments(text):
    text = str(text)
    if 'شقتين' in text or 'مع شقتين' in text or 'وشقتين' in text: return 2
    elif 'شقة' in text or 'مع شقة' in text or 'وشقة' in text: return 1
    return 0
df['extra_apartments'] = norm_text.apply(extract_extra_apartments)

def extract_master_bedrooms(t):
    m = re.search(r'(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)', t)
    if m:
        c = int(m.group(1))
        if 1 <= c <= 8: return float(c)
    if 'ماستر' in t or 'جناح رئيسي' in t: return 1.0
    return 0.0
df['master_bedrooms'] = norm_text.apply(extract_master_bedrooms)

def extract_living_rooms(t):
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
df['living_rooms'] = norm_text.apply(extract_living_rooms)

def extract_facade(t):
    if re.search(r'شارعين|زاوية|زاويه|رأس بلك|راس بلك|ثلاث شوارع', t): return 'corner'
    elif re.search(r'شمالي|شماليه|شمالية', t): return 'north'
    elif re.search(r'شرقي|شرقيه|شرقية', t): return 'east'
    elif re.search(r'جنوبي|جنوبيه|جنوبية', t): return 'south'
    elif re.search(r'غربي|غربيه|غربية', t): return 'west'
    return 'unknown'
df['facade'] = norm_text.apply(extract_facade)

df['has_double_height'] = norm_text.str.contains('دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع', regex=True).astype(int)
df['elevator_installed'] = norm_text.str.contains('مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب', regex=True).astype(int)
df['elevator_ready'] = norm_text.str.contains('تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد', regex=True).astype(int)
df['has_custom_built'] = norm_text.str.contains('بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام', regex=True).astype(int)
df['has_spacious_yard'] = norm_text.str.contains('حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه', regex=True).astype(int)
df['is_modern'] = norm_text.str.contains('مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن', regex=True).astype(int)

facade_dummies = pd.get_dummies(df['facade'], prefix='facade', drop_first=True, dtype=int)
df = pd.concat([df, facade_dummies], axis=1)

clean_df = df[
    (df['price'] >= 150000) &
    (df['price'] <= 7000000) &
    (df['size'] >= 100) &
    (df['size'] <= 1200)
].copy()

# 2. Attach external benchmark anchors (Zero Leakage from target)
mapper = AqarBenchmarkMapper(os.path.join(BASE_DIR, "aqar_benchmarks_2026.json"))
bm_records = [mapper.get_benchmark(d, s) for d, s in zip(clean_df["neighborhood"], clean_df["size"])]
clean_df["base_hist"] = [b["hist_price_base"] for b in bm_records]
clean_df["base_2026"] = [b["avg_price_2026"] for b in bm_records]
clean_df["imputed"] = [b["is_imputed"] for b in bm_records]

# 3. Target: log(price / base_hist)
y_rel = np.log(clean_df["price"] / clean_df["base_hist"])
y_orig = clean_df["price"]

facade_cols = [c for c in clean_df.columns if c.startswith('facade_')]
base_feature_cols = [
    'size', 'bedrooms', 'bathrooms', 'is_villa', 'is_apartment',
    'has_driver_room', 'has_maid_room', 'is_new', 'has_elevator',
    'has_pool', 'has_annex', 'has_garage', 'has_central_ac',
    'has_duplex_stairs', 'street_width', 'extra_apartments',
    'master_bedrooms', 'living_rooms', 'has_double_height',
    'elevator_installed', 'elevator_ready', 'has_custom_built',
    'has_spacious_yard', 'is_modern'
]
feature_cols = base_feature_cols + facade_cols
feats_with_nh = ["neighborhood"] + feature_cols

X = clean_df[feats_with_nh + ["base_hist", "base_2026", "imputed"]]

X_tr, X_te, y_tr, y_te, y_orig_tr, y_orig_te = train_test_split(
    X, y_rel, y_orig, test_size=0.20, random_state=42
)

print(f"Train size: {X_tr.shape[0]} | Test size: {X_te.shape[0]}")

# 4. Pipeline with ColumnTransformer and TargetEncoder
pipe_relative = Pipeline([
    ("prep", ColumnTransformer([
        ("te", TargetEncoder(cv=5, random_state=42, smooth="auto"), ["neighborhood"])
    ], remainder="passthrough")),
    ("gbr", GradientBoostingRegressor(
        n_estimators=160,
        max_depth=4,
        min_samples_leaf=12,
        min_samples_split=24,
        subsample=0.85,
        max_features=0.85,
        learning_rate=0.05,
        random_state=42
    ))
])

pipe_relative.fit(X_tr[feats_with_nh], y_tr)

# 5. Internal Holdout Evaluation (on historical price level)
pred_ratio_te = pipe_relative.predict(X_te[feats_with_nh])
pred_hist_te = X_te["base_hist"] * np.exp(pred_ratio_te)

mape_hist = mean_absolute_percentage_error(y_orig_te, pred_hist_te) * 100
mae_hist = mean_absolute_error(y_orig_te, pred_hist_te)
r2_hist = r2_score(y_orig_te, pred_hist_te)

print("\n--- RELATIVE RATIO MODEL INTERNAL HOLDOUT PERFORMANCE ---")
print(f"R² on price: {r2_hist:.4f}")
print(f"MAE: {mae_hist:,.0f} SAR")
print(f"MAPE: {mape_hist:.2f}%")

# Segmented by imputed vs non-imputed benchmark districts
mask_direct = ~X_te["imputed"]
mape_direct = mean_absolute_percentage_error(y_orig_te[mask_direct], pred_hist_te[mask_direct]) * 100
mape_imputed = mean_absolute_percentage_error(y_orig_te[~mask_direct], pred_hist_te[~mask_direct]) * 100
print(f"MAPE on 8 Key Benchmark Districts (direct): {mape_direct:.2f}% (N={mask_direct.sum()})")
print(f"MAPE on City Median Imputed Districts: {mape_imputed:.2f}% (N={(~mask_direct).sum()})")

# 6. Save the Relative Ratio Model Pipeline
rel_pipeline_path = os.path.join(BASE_DIR, "house_price_relative_pipeline.joblib")
joblib.dump({
    "pipe": pipe_relative,
    "feature_names": feats_with_nh,
    "numeric_features": feature_cols,
    "facade_columns": facade_cols,
    "benchmark_mapper": mapper
}, rel_pipeline_path)
print(f"Saved relative pipeline to: {rel_pipeline_path}")
