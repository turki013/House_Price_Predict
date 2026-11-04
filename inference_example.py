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

# Load trained pipeline bundle
pipeline_bundle = joblib.load(os.path.join(BASE_DIR, "house_price_pipeline.joblib"))
encoder = pipeline_bundle["target_encoder"]
model = pipeline_bundle["model"]
feature_names = pipeline_bundle["feature_names"]
benchmark_mapper = pipeline_bundle.get("benchmark_mapper") or AqarBenchmarkMapper()

arabic_digits = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9'}
def normalize_text(s):
    for k, v in arabic_digits.items():
        s = s.replace(k, v)
    return s

def extract_features_from_listing(listing):
    full_text = normalize_text(listing.get('details', '') + ' ' + listing.get('listTitle', ''))
    
    # 1. Master bedrooms
    def extract_master(t):
        m = re.search(r'(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)', t)
        if m:
            c = int(m.group(1))
            if 1 <= c <= 8: return float(c)
        if 'ماستر' in t or 'جناح رئيسي' in t: return 1.0
        return 0.0

    # 2. Living rooms
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

    # 3. Facade
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
        return float(listing.get('street_width', 15.0))

    def extract_extra_apts(t):
        if 'شقتين' in t or 'مع شقتين' in t or 'وشقتين' in t: return 2
        elif 'شقة' in t or 'مع شقة' in t or 'وشقة' in t: return 1
        return int(listing.get('extra_apartments', 0))

    facade_val = listing.get('facade') or extract_facade(full_text)

    row = {
        'size': float(listing['size']),
        'bedrooms': float(listing.get('bedrooms', 4)),
        'bathrooms': float(listing.get('bathrooms', 4)),
        'is_villa': 1 if 'فيلا' in full_text or listing.get('is_villa', 1) else 0,
        'is_apartment': 1 if ('شقة' in full_text or 'شقه' in full_text) and 'فيلا' not in full_text else 0,
        'has_driver_room': int(bool(re.search(r'سائق|سواق', full_text)) or listing.get('has_driver_room', 0)),
        'has_maid_room': int(bool(re.search(r'خادمة|شغالة', full_text)) or listing.get('has_maid_room', 0)),
        'is_new': int(bool(re.search(r'جديد|جديدة|مشروع', full_text)) or listing.get('is_new', 1)),
        'has_elevator': int(bool(re.search(r'مصعد|اصنصير', full_text)) or listing.get('has_elevator', 0) or listing.get('elevator_installed', 0)),
        'has_pool': int(bool(re.search(r'مسبح', full_text)) or listing.get('has_pool', 0)),
        'has_annex': int(bool(re.search(r'ملحق', full_text)) or listing.get('has_annex', 0)),
        'has_garage': int(bool(re.search(r'مدخل سيارة|مدخل سياره|كراج', full_text)) or listing.get('has_garage', 1)),
        'has_central_ac': int(bool(re.search(r'تكييف مخفي|تكيف مخفي|مركزي|سبلت مخفي', full_text)) or listing.get('has_central_ac', 0)),
        'has_duplex_stairs': int(bool(re.search(r'درج صالة|درج صاله|درج داخلي|دوبلكس', full_text)) or listing.get('has_duplex_stairs', 0)),
        'street_width': extract_street_width(full_text),
        'extra_apartments': extract_extra_apts(full_text),
        'master_bedrooms': float(listing['master_bedrooms']) if 'master_bedrooms' in listing else extract_master(full_text),
        'living_rooms': float(listing['living_rooms']) if 'living_rooms' in listing else extract_living(full_text),
        'has_double_height': int(bool(re.search(r'دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع', full_text)) or listing.get('has_double_height', 0)),
        'elevator_installed': int(bool(re.search(r'مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب', full_text)) or listing.get('elevator_installed', 0)),
        'elevator_ready': int(bool(re.search(r'تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد', full_text)) or listing.get('elevator_ready', 0)),
        'has_custom_built': int(bool(re.search(r'بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام', full_text)) or listing.get('has_custom_built', 0)),
        'has_spacious_yard': int(bool(re.search(r'حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه', full_text)) or listing.get('has_spacious_yard', 1)),
        'is_modern': int(bool(re.search(r'مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن', full_text)) or listing.get('is_modern', 1)),
        'facade_corner': 1 if facade_val == 'corner' else 0,
        'facade_east': 1 if facade_val == 'east' else 0,
        'facade_north': 1 if facade_val == 'north' else 0,
        'facade_south': 1 if facade_val == 'south' else 0,
        'facade_unknown': 1 if facade_val == 'unknown' else 0,
        'facade_west': 1 if facade_val == 'west' else 0,
        'neighborhood': benchmark_mapper.clean_district(listing['neighborhood'])
    }
    return row

def predict_listing(listing):
    features = extract_features_from_listing(listing)
    df_sample = pd.DataFrame([features])
    
    # Target encode neighborhood
    df_sample['neighborhood_encoded'] = encoder.transform(df_sample[['neighborhood']])
    
    # Predict log-scale and transform to base historical SAR
    pred_log = model.predict(df_sample[feature_names])
    pred_hist_sar = float(np.expm1(pred_log)[0])
    
    # Dynamic 2026 calibration via benchmark mapper
    pred_2026_sar, bm_info = benchmark_mapper.calibrate(pred_hist_sar, listing['neighborhood'], listing['size'])
    
    return {
        "features": features,
        "pred_hist_sar": pred_hist_sar,
        "pred_2026_sar": pred_2026_sar,
        "benchmark_info": bm_info
    }

# --- TEST CASE 1: AL-NARJIS MODERN VILLA ---
al_narjis_listing = {
    'listTitle': 'فيلا للبيع في شارع أبي زكريا بن منصور, حي النرجس, مدينة الرياض (مشروع فلل إيثار)',
    'details': 'فلل إيثار | حي النرجس فرصة مميزة لامتلاك فيلا بتصميم عصري في موقع استراتيجي شمال الرياض تشطيبات راقية 4 غرف نوم ماستر مجلس وصالة بمساحات واسعة غرفة خادمة غرفة سائق مصعد راكب دبل هايت تكييف مخفي واجهات شمالي مساحة 200م',
    'size': 200.0,
    'bedrooms': 5,
    'bathrooms': 5,
    'neighborhood': 'النرجس',
    'street_width': 20.0,
    'actual_price': 2540000.0,
    'actual_range': '2.45M – 2.63M SAR'
}

# --- TEST CASE 2: AL-WURUD RESIDENTIAL VILLA WITH APARTMENT ---
al_wurud_listing = {
    'listTitle': 'فيلا للبيع في حي الورود, الرياض مع شقة مستقلة استثمارية',
    'details': 'فيلا مميزة في حي الورود وسط الرياض موقع هادئ قريب من طريق الملك عبدالله والملك فهد، تتكون من دورين وشقة مستقلة بمدخل خاص مؤجرة، 4 غرف نوم ماستر، مجلس رجال ومجلس نساء وصالة عائلية، غرفة خادمة وغرفة سائق، حوش ومدخل سيارة، شارع 15م، بناء شخصي.',
    'size': 245.0,
    'bedrooms': 4,
    'bathrooms': 4,
    'neighborhood': 'الورود',
    'street_width': 15.0,
    'extra_apartments': 1,
    'actual_price': 2900000.0,
    'actual_range': '~2.90M SAR'
}

cases = [
    ("Al-Narjis Modern Luxury Villa", al_narjis_listing),
    ("Al-Wurud Villa with Apartment", al_wurud_listing)
]

print("=" * 90)
print("PRODUCTION INFERENCE VERIFICATION: EMPIRICAL 2026 BENCHMARK MAPPING (SMOOTHED 2015-16 BASE)")
print("=" * 90)

table_rows = []
for name, listing in cases:
    res = predict_listing(listing)
    bm = res["benchmark_info"]
    diff_sar = res["pred_2026_sar"] - listing['actual_price']
    diff_pct = (diff_sar / listing['actual_price']) * 100
    
    print(f"\n▶ Case: {name}")
    print(f"  District: {listing['neighborhood']} | Size: {listing['size']} m² | Size Bucket: {bm['size_bucket']}")
    print(f"  Aqar Benchmark 2026 Avg: {bm['avg_price_2026']:,.0f} SAR | Historical 2015-16 Avg: {bm['hist_price_base']:,.0f} SAR")
    print(f"  Empirical Growth Ratio: {bm['growth_ratio']:.4f}x (Imputed: {bm['is_imputed']})")
    print(f"  Historical Base Model Valuation: {res['pred_hist_sar']:,.2f} SAR")
    print(f"  Calibrated 2026 Market Valuation: {res['pred_2026_sar']:,.2f} SAR")
    print(f"  Live Listing Price: {listing['actual_price']:,.2f} SAR ({listing['actual_range']})")
    print(f"  Pricing Error to Live Market: {diff_pct:+.2f}% ({diff_sar:+,.0f} SAR)")
    
    table_rows.append({
        "Property Case": name,
        "District": listing['neighborhood'],
        "Size": f"{listing['size']} m²",
        "Empirical Growth Index": f"{bm['growth_ratio']:.4f}x",
        "Historical Base Valuation": f"{res['pred_hist_sar']:,.0f} SAR",
        "Calibrated 2026 Valuation": f"{res['pred_2026_sar']:,.0f} SAR",
        "Actual Listing Price": f"{listing['actual_price']:,.0f} SAR",
        "Variance (%)": f"{diff_pct:+.2f}%"
    })

print("\n" + "=" * 90)
print("FINAL VALUATION COMPARISON TABLE")
print("=" * 90)
df_table = pd.DataFrame(table_rows)
print(df_table.to_string(index=False))
