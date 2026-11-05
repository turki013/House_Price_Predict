"""Parity test: the unified extractor must reproduce the ORIGINAL training features.

``legacy_features`` below is a frozen copy of the old vectorised training code
(details-only routing for driver/maid/is_new, etc.). The live-style path
(dict in -> ``extract_features``) is checked against it row by row.

Run:  python -m pytest tests/test_features.py   (or)   python tests/test_features.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import DATA_PATH, FEATURE_COLS, FACADES, clean_district, extract_features  # noqa: E402

_DIG = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}


def _norm(s):
    for k, v in _DIG.items():
        s = s.replace(k, v)
    return s


def legacy_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["listTitle"] = df["listTitle"].fillna("")
    df["details"] = df["details"].fillna("")
    n = (df["details"] + " " + df["listTitle"]).apply(_norm)
    c = lambda s, pat: s.str.contains(pat, regex=True, na=False).astype(int)  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["size"], out["bedrooms"], out["bathrooms"] = df["size"], df["bedrooms"], df["bathrooms"]
    out["is_villa"] = (c(df["listTitle"], "فيلا") | c(df["details"], "فيلا")).astype(int)
    out["is_apartment"] = (c(df["listTitle"], "شقة|شقه") | c(df["details"], "شقة|شقه")).astype(int)
    out["has_driver_room"] = c(df["details"], "سائق|سواق")
    out["has_maid_room"] = c(df["details"], "خادمة|شغالة")
    out["is_new"] = c(df["details"], "جديد|جديدة")
    out["has_elevator"] = c(n, "مصعد|اصنصير")
    out["has_pool"] = c(n, "مسبح")
    out["has_annex"] = c(n, "ملحق")
    out["has_garage"] = c(n, "مدخل سيارة|مدخل سياره|كراج")
    out["has_central_ac"] = c(n, "تكييف مخفي|تكيف مخفي|مركزي|سبلت مخفي")
    out["has_duplex_stairs"] = c(n, "درج صالة|درج صاله|درج داخلي|دوبلكس")

    def sw(t):
        m = re.search(r"(?:شارع|عرض)\s*(\d{2,3})", str(t))
        return float(int(m.group(1))) if m and 8 <= int(m.group(1)) <= 100 else 15.0

    def ea(t):
        return 2 if ("شقتين" in t or "مع شقتين" in t or "وشقتين" in t) else (
            1 if ("شقة" in t or "مع شقة" in t or "وشقة" in t) else 0)

    def mb(t):
        m = re.search(r"(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)", t)
        if m and 1 <= int(m.group(1)) <= 8:
            return float(int(m.group(1)))
        return 1.0 if ("ماستر" in t or "جناح رئيسي" in t) else 0.0

    def lv(t):
        k = 0
        m = re.search(r"(\d+)\s*مجالس", t)
        k += int(m.group(1)) if m else (2 if "مجلسين" in t else (1 if "مجلس" in t else 0))
        m = re.search(r"(\d+)\s*صالات", t)
        k += int(m.group(1)) if m else (2 if "صالتين" in t else (1 if ("صالة" in t or "صاله" in t) else 0))
        return float(min(k, 8))

    def fc(t):
        if re.search("شارعين|زاوية|زاويه|رأس بلك|راس بلك|ثلاث شوارع", t): return "corner"
        if re.search("شمالي|شماليه|شمالية", t): return "north"
        if re.search("شرقي|شرقيه|شرقية", t): return "east"
        if re.search("جنوبي|جنوبيه|جنوبية", t): return "south"
        if re.search("غربي|غربيه|غربية", t): return "west"
        return "unknown"

    out["street_width"] = n.apply(sw)
    out["extra_apartments"] = n.apply(ea)
    out["master_bedrooms"] = n.apply(mb)
    out["living_rooms"] = n.apply(lv)
    out["has_double_height"] = c(n, "دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع")
    out["elevator_installed"] = c(n, "مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب")
    out["elevator_ready"] = c(n, "تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد")
    out["has_custom_built"] = c(n, "بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام")
    out["has_spacious_yard"] = c(n, "حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه")
    out["is_modern"] = c(n, "مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن")
    facade = n.apply(fc)
    for f in FACADES:
        out[f"facade_{f}"] = (facade == f).astype(int)
    return out[FEATURE_COLS]


def _sample(n=1500):
    return pd.read_csv(DATA_PATH).sample(n=n, random_state=0)


def test_live_style_path_matches_legacy_training_features():
    sample = _sample()
    items = [
        {"details": r["details"], "title": r["listTitle"], "size": r["size"],
         "bedrooms": r["bedrooms"], "bathrooms": r["bathrooms"], "district": r["location"]}
        for r in sample.to_dict("records")
    ]
    live_style = pd.DataFrame([extract_features(**i) for i in items], index=sample.index)
    legacy = legacy_features(sample)
    diff = (live_style[FEATURE_COLS].astype(float).values != legacy[FEATURE_COLS].astype(float).values)
    both_nan = np.isnan(live_style[FEATURE_COLS].astype(float).values) & np.isnan(legacy.astype(float).values)
    bad = diff & ~both_nan
    assert not bad.any(), {c: int(bad[:, i].sum()) for i, c in enumerate(FEATURE_COLS) if bad[:, i].any()}


def test_details_only_features_ignore_title():
    base = dict(details="", size=300, bedrooms=4, bathrooms=4, district="حي النرجس - الرياض")
    row = extract_features(title="مشروع جديد خادمة سائق", **base)
    assert row["is_new"] == 0 and row["has_maid_room"] == 0 and row["has_driver_room"] == 0
    row = extract_features(title="", **{**base, "details": "جديد خادمة سائق"})
    assert row["is_new"] == 1 and row["has_maid_room"] == 1 and row["has_driver_room"] == 1


def test_project_word_does_not_set_is_new():
    row = extract_features(details="مشروع فلل", title="", size=250, bedrooms=4, bathrooms=4, district="النرجس")
    assert row["is_new"] == 0


def test_district_cleaning_is_shared():
    assert clean_district("حي النرجس - الرياض") == "النرجس"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
