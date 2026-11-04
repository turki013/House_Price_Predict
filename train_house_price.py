import re
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Reconfigure stdout for utf-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else r"C:\Users\HUAWEI\House_price"
sys.path.append(BASE_DIR)
from aqar_benchmark_mapper import AqarBenchmarkMapper

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from xgboost import XGBRegressor

RANDOM_STATE = 42
DATA_PATH = os.path.join(BASE_DIR, "Aqar_data.csv")
BENCHMARK_PATH = os.path.join(BASE_DIR, "aqar_benchmarks_2026.json")
OUTPUT_DIR = BASE_DIR

print("--- 1. DATA AUDIT & LUXURY FEATURE EXTRACTION ---")
df_raw = pd.read_csv(DATA_PATH)
print(f"Initial raw records: {df_raw.shape[0]}")

# 1. Feature Extraction from raw text
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

# Base property types & amenities
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
        if 8 <= w <= 100:
            return float(w)
    return 15.0

df['street_width'] = norm_text.apply(extract_street_width)

def extract_extra_apartments(text):
    text = str(text)
    if 'شقتين' in text or 'مع شقتين' in text or 'وشقتين' in text:
        return 2
    elif 'شقة' in text or 'مع شقة' in text or 'وشقة' in text:
        return 1
    return 0

df['extra_apartments'] = norm_text.apply(extract_extra_apartments)

# Advanced Luxury Signals (Regex Extraction)
def extract_master_bedrooms(t):
    m = re.search(r'(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)', t)
    if m:
        c = int(m.group(1))
        if 1 <= c <= 8:
            return float(c)
    if 'ماستر' in t or 'جناح رئيسي' in t:
        return 1.0
    return 0.0

df['master_bedrooms'] = norm_text.apply(extract_master_bedrooms)

def extract_living_rooms(t):
    count = 0
    m_majlis = re.search(r'(\d+)\s*مجالس', t)
    if m_majlis:
        count += int(m_majlis.group(1))
    elif 'مجلسين' in t:
        count += 2
    elif 'مجلس' in t:
        count += 1
        
    m_sala = re.search(r'(\d+)\s*صالات', t)
    if m_sala:
        count += int(m_sala.group(1))
    elif 'صالتين' in t:
        count += 2
    elif 'صالة' in t or 'صاله' in t:
        count += 1
    return float(min(count, 8))

df['living_rooms'] = norm_text.apply(extract_living_rooms)

def extract_facade(t):
    if re.search(r'شارعين|زاوية|زاويه|رأس بلك|راس بلك|ثلاث شوارع', t):
        return 'corner'
    elif re.search(r'شمالي|شماليه|شمالية', t):
        return 'north'
    elif re.search(r'شرقي|شرقيه|شرقية', t):
        return 'east'
    elif re.search(r'جنوبي|جنوبيه|جنوبية', t):
        return 'south'
    elif re.search(r'غربي|غربيه|غربية', t):
        return 'west'
    return 'unknown'

df['facade'] = norm_text.apply(extract_facade)

df['has_double_height'] = norm_text.str.contains('دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع', regex=True).astype(int)
df['elevator_installed'] = norm_text.str.contains('مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب', regex=True).astype(int)
df['elevator_ready'] = norm_text.str.contains('تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد', regex=True).astype(int)
df['has_custom_built'] = norm_text.str.contains('بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام', regex=True).astype(int)
df['has_spacious_yard'] = norm_text.str.contains('حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه', regex=True).astype(int)
df['is_modern'] = norm_text.str.contains('مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن', regex=True).astype(int)

# Facade dummy encoding
facade_dummies = pd.get_dummies(df['facade'], prefix='facade', drop_first=True, dtype=int)
df = pd.concat([df, facade_dummies], axis=1)

# Clean Outliers via strict residential domain bounds
clean_df = df[
    (df['price'] >= 150000) &
    (df['price'] <= 7000000) &
    (df['size'] >= 100) &
    (df['size'] <= 1200)
].copy()
print(f"Cleaned records post-filtering: {clean_df.shape[0]} (removed {df.shape[0] - clean_df.shape[0]} outliers)")

# Assign benchmark mapping to clean dataset
benchmark_mapper = AqarBenchmarkMapper(BENCHMARK_PATH)
clean_df = benchmark_mapper.assign_benchmarks(clean_df, district_col='neighborhood', size_col='size')
print(f"Assigned verified 2026 benchmarks (smoothed 2015-2016 baseline). Direct coverage: {((~clean_df['benchmark_imputed']).mean()*100):.1f}%, remainder robustly median imputed.")

# Target Skewness & Log Transformation
raw_skew = clean_df['price'].skew()
log_skew = np.log1p(clean_df['price']).skew()
print(f"Price skewness: raw = {raw_skew:.2f} -> log1p = {log_skew:.2f}")

drop_metadata_cols = ['price', 'listTitle', 'details', 'location', 'time', 'facade', 'benchmark_2026', 'benchmark_hist', 'growth_ratio', 'benchmark_imputed']
X = clean_df.drop(columns=drop_metadata_cols)
y = np.log1p(clean_df['price'])
y_orig = clean_df['price']

# --- 2. TRAIN / TEST SPLIT (SPLIT FIRST DISCIPLINE) ---
X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
    X, y, y_orig, test_size=0.20, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# --- 3. LEAK-FREE TARGET ENCODING (OUT-OF-FOLD) ---
cv_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
te_enc = TargetEncoder(smooth='auto', cv=cv_kf)
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
X_train_encoded['neighborhood_encoded'] = te_enc.fit_transform(X_train[['neighborhood']], y_train)
X_test_encoded['neighborhood_encoded'] = te_enc.transform(X_test[['neighborhood']])

feature_cols = [c for c in X_train_encoded.columns if c != 'neighborhood']

# --- 4. BASELINE VS REGULARIZED MODEL EXPERIMENTS ---
# 1. Dummy Baseline
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_encoded[feature_cols], y_train)

# 2. Simple Interpretable Baseline: Ridge
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_encoded[feature_cols])
X_te_sc = scaler.transform(X_test_encoded[feature_cols])
ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
ridge.fit(X_tr_sc, y_train)

# 3. Regularized GradientBoostingRegressor (Champion)
reg_gbr = GradientBoostingRegressor(
    n_estimators=160,
    max_depth=4,
    min_samples_leaf=12,
    min_samples_split=24,
    subsample=0.85,
    max_features=0.85,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)
reg_gbr.fit(X_train_encoded[feature_cols], y_train)

# 4. Regularized XGBoost
reg_xgb = XGBRegressor(
    n_estimators=140,
    max_depth=4,
    min_child_weight=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=2.0,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)
reg_xgb.fit(X_train_encoded[feature_cols], y_train)

# --- 5. EVALUATION ---
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, y_orig_te, is_scaled=False):
    x_tr_eval = X_tr_sc if is_scaled else X_tr
    x_te_eval = X_te_sc if is_scaled else X_te
    
    tr_r2 = model.score(x_tr_eval, y_tr)
    te_r2 = model.score(x_te_eval, y_te)
    gap = (tr_r2 - te_r2) * 100
    
    pred_log = model.predict(x_te_eval)
    pred_sar = np.expm1(pred_log)
    
    rmse_log = np.sqrt(mean_squared_error(y_te, pred_log))
    mae_sar = mean_absolute_error(y_orig_te, pred_sar)
    rmse_sar = np.sqrt(mean_squared_error(y_orig_te, pred_sar))
    mape = mean_absolute_percentage_error(y_orig_te, pred_sar) * 100
    
    return {
        "Model": name,
        "Train_R2": round(tr_r2, 4),
        "Test_R2": round(te_r2, 4),
        "Gap_pct": round(gap, 2),
        "RMSE_log": round(rmse_log, 4),
        "MAE_SAR": round(mae_sar, 0),
        "RMSE_SAR": round(rmse_sar, 0),
        "MAPE_pct": round(mape, 2)
    }

results = [
    evaluate_model("Dummy (Mean)", dummy, X_train_encoded[feature_cols], y_train, X_test_encoded[feature_cols], y_test, y_orig_test),
    evaluate_model("Ridge (alpha=10)", ridge, X_train_encoded[feature_cols], y_train, X_test_encoded[feature_cols], y_test, y_orig_test, is_scaled=True),
    evaluate_model("Regularized GBR (depth=4)", reg_gbr, X_train_encoded[feature_cols], y_train, X_test_encoded[feature_cols], y_test, y_orig_test),
    evaluate_model("Regularized XGBoost", reg_xgb, X_train_encoded[feature_cols], y_train, X_test_encoded[feature_cols], y_test, y_orig_test)
]

res_df = pd.DataFrame(results)
print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(res_df.to_string(index=False))

# Feature importances
importances = reg_gbr.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n--- TOP FEATURE IMPORTANCES (REGULARIZED GBR) ---")
print(feat_imp.head(15).to_string(index=False))

# --- 6. SERIALIZATION & EXPORT ---
artifacts = {
    "target_encoder": te_enc,
    "model": reg_gbr,
    "feature_names": feature_cols,
    "facade_columns": [c for c in feature_cols if c.startswith('facade_')],
    "neighborhoods_seen": [str(c) for c in te_enc.categories_[0]] if hasattr(te_enc, 'categories_') and te_enc.categories_ is not None else list(X_train['neighborhood'].unique()),
    "benchmark_mapper": benchmark_mapper
}
pipeline_path = os.path.join(OUTPUT_DIR, "house_price_pipeline.joblib")
joblib.dump(artifacts, pipeline_path)
print(f"\nPipeline serialized to: {pipeline_path}")

metadata = {
    "model_name": "riyadh_house_price_luxury_enhanced_pipeline",
    "target": "price",
    "target_transform": "np.log1p",
    "baseline_epoch": "2015-2016 (smoothed half-years 1_2015 to 2_2016)",
    "calibration_target": "2026 H1 (1_2026)",
    "features": feature_cols,
    "clean_bounds": {
        "min_price": 150000,
        "max_price": 7000000,
        "min_size": 100,
        "max_size": 1200
    },
    "metrics_comparison": results,
    "top_features": feat_imp.to_dict(orient="records"),
    "benchmarks_used": BENCHMARK_PATH,
    "environment": {
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "xgboost": sys.modules['xgboost'].__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__
    }
}

metadata_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadata saved to: {metadata_path}")
