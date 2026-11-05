"""Single source of truth for feature extraction.

Training, live evaluation and inference all call ``extract_features`` so the
feature definitions cannot drift apart. Some features are deliberately computed
on ``details`` only and others on ``details + title``; that routing was the
original training definition and is preserved here exactly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Aqar_data.csv"

PRICE_BOUNDS = (150_000, 7_000_000)
SIZE_BOUNDS = (100, 1_200)
DEDUP_KEY = ["location", "size", "price", "bedrooms", "bathrooms"]

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

FACADES = ("corner", "east", "north", "south", "unknown", "west")

BINARY_FEATURES = [
    "is_villa", "is_apartment", "has_driver_room", "has_maid_room", "is_new",
    "has_elevator", "has_pool", "has_annex", "has_garage", "has_central_ac",
    "has_duplex_stairs", "has_double_height", "elevator_installed",
    "elevator_ready", "has_custom_built", "has_spacious_yard", "is_modern",
]
NUMERIC_FEATURES = [
    "size", "bedrooms", "bathrooms", "street_width", "extra_apartments",
    "master_bedrooms", "living_rooms",
]
FACADE_FEATURES = [f"facade_{f}" for f in FACADES]
FEATURE_COLS = NUMERIC_FEATURES + BINARY_FEATURES + FACADE_FEATURES
CATEGORICAL_COL = "neighborhood"
MODEL_INPUT_COLS = [CATEGORICAL_COL] + FEATURE_COLS

_R = {
    "villa": re.compile("فيلا"),
    "apartment": re.compile("شقة|شقه"),
    "driver": re.compile("سائق|سواق"),
    "maid": re.compile("خادمة|شغالة"),
    "new": re.compile("جديد|جديدة"),
    "elevator": re.compile("مصعد|اصنصير"),
    "pool": re.compile("مسبح"),
    "annex": re.compile("ملحق"),
    "garage": re.compile("مدخل سيارة|مدخل سياره|كراج"),
    "central_ac": re.compile("تكييف مخفي|تكيف مخفي|مركزي|سبلت مخفي"),
    "duplex": re.compile("درج صالة|درج صاله|درج داخلي|دوبلكس"),
    "double_height": re.compile("دبل هايت|دبلهايت|اسقف عاليه|أسقف عالية|سقف عالي|ارتفاع عالي|سقف مرتفع"),
    "elevator_installed": re.compile("مصعد راكب|راكب مصعد|اصنصير|مصعد شغال|المصعد راكب"),
    "elevator_ready": re.compile("تأسيس مصعد|تاسيس مصعد|مؤسس مصعد|مكان مصعد"),
    "custom_built": re.compile("بناء شخصي|إشراف هندسي|اشراف هندسي|كود سعودي|تأمين ملاذ|شهادة إتمام"),
    "yard": re.compile("حوش|فناء|حديقة|ارتداد واسع|مسطحات خضراء|جلسة خارجية|جلسه خارجيه"),
    "modern": re.compile("مودرن|عصري|تصميم حديث|تصميم عصري|فيلا مودرن"),
    "street_width": re.compile(r"(?:شارع|عرض)\s*(\d{2,3})"),
    "master_count": re.compile(r"(\d+)\s*(?:غرف(?:ة)?\s*(?:نوم\s*)?)?(?:ماستر|أجنحة|اجنحه)"),
    "majlis_count": re.compile(r"(\d+)\s*مجالس"),
    "sala_count": re.compile(r"(\d+)\s*صالات"),
    "f_corner": re.compile("شارعين|زاوية|زاويه|رأس بلك|راس بلك|ثلاث شوارع"),
    "f_north": re.compile("شمالي|شماليه|شمالية"),
    "f_east": re.compile("شرقي|شرقيه|شرقية"),
    "f_south": re.compile("جنوبي|جنوبيه|جنوبية"),
    "f_west": re.compile("غربي|غربيه|غربية"),
}


def normalize_text(s: str) -> str:
    return str(s).translate(_ARABIC_DIGITS)


def clean_district(location: str) -> str:
    """'حي النرجس - الرياض' -> 'النرجس' (the training-time rule)."""
    return str(location).replace("حي", "").replace("- الرياض", "").strip()


def _txt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _street_width(t: str) -> float:
    m = _R["street_width"].search(t)
    if m and 8 <= int(m.group(1)) <= 100:
        return float(int(m.group(1)))
    return 15.0


def _extra_apartments(t: str) -> int:
    if "شقتين" in t:
        return 2
    if "شقة" in t:
        return 1
    return 0


def _master_bedrooms(t: str) -> float:
    m = _R["master_count"].search(t)
    if m and 1 <= int(m.group(1)) <= 8:
        return float(int(m.group(1)))
    return 1.0 if ("ماستر" in t or "جناح رئيسي" in t) else 0.0


def _living_rooms(t: str) -> float:
    count = 0
    m = _R["majlis_count"].search(t)
    if m:
        count += int(m.group(1))
    elif "مجلسين" in t:
        count += 2
    elif "مجلس" in t:
        count += 1
    m = _R["sala_count"].search(t)
    if m:
        count += int(m.group(1))
    elif "صالتين" in t:
        count += 2
    elif "صالة" in t or "صاله" in t:
        count += 1
    return float(min(count, 8))


def _facade(t: str) -> str:
    for key, name in (("f_corner", "corner"), ("f_north", "north"), ("f_east", "east"),
                      ("f_south", "south"), ("f_west", "west")):
        if _R[key].search(t):
            return name
    return "unknown"


def extract_features(*, details, title, size, bedrooms, bathrooms, district) -> dict:
    """Features for ONE listing. ``bedrooms``/``bathrooms`` may be NaN; the caller
    decides how to impute (nothing is filled in silently here)."""
    details, title = _txt(details), _txt(title)
    norm = normalize_text(details + " " + title)
    has = lambda key, text: int(bool(_R[key].search(text)))  # noqa: E731

    row = {
        "size": float(size),
        "bedrooms": float(bedrooms) if bedrooms is not None else np.nan,
        "bathrooms": float(bathrooms) if bathrooms is not None else np.nan,
        # --- routed on title OR details (raw text) ---
        "is_villa": int(bool(_R["villa"].search(title) or _R["villa"].search(details))),
        "is_apartment": int(bool(_R["apartment"].search(title) or _R["apartment"].search(details))),
        # --- routed on details ONLY (training definition) ---
        "has_driver_room": has("driver", details),
        "has_maid_room": has("maid", details),
        "is_new": has("new", details),
        # --- routed on details + title, digits normalised ---
        "has_elevator": has("elevator", norm),
        "has_pool": has("pool", norm),
        "has_annex": has("annex", norm),
        "has_garage": has("garage", norm),
        "has_central_ac": has("central_ac", norm),
        "has_duplex_stairs": has("duplex", norm),
        "has_double_height": has("double_height", norm),
        "elevator_installed": has("elevator_installed", norm),
        "elevator_ready": has("elevator_ready", norm),
        "has_custom_built": has("custom_built", norm),
        "has_spacious_yard": has("yard", norm),
        "is_modern": has("modern", norm),
        "street_width": _street_width(norm),
        "extra_apartments": _extra_apartments(norm),
        "master_bedrooms": _master_bedrooms(norm),
        "living_rooms": _living_rooms(norm),
        CATEGORICAL_COL: clean_district(district),
    }
    facade = _facade(norm)
    for f in FACADES:
        row[f"facade_{f}"] = int(facade == f)
    return row


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-wise application of ``extract_features`` to a raw Aqar_data frame."""
    cols = ["details", "listTitle", "size", "bedrooms", "bathrooms", "location"]
    rows = [
        extract_features(details=r["details"], title=r["listTitle"], size=r["size"],
                         bedrooms=r["bedrooms"], bathrooms=r["bathrooms"], district=r["location"])
        for r in raw[cols].to_dict("records")
    ]
    return pd.DataFrame(rows, index=raw.index)[MODEL_INPUT_COLS]


def load_training_frame(path=DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """Raw CSV -> de-duplicated, bounded frame with ``price`` + model inputs."""
    raw = pd.read_csv(path)
    n0 = len(raw)
    raw = raw.drop_duplicates(subset=DEDUP_KEY)
    n1 = len(raw)
    feats = build_feature_frame(raw)
    frame = pd.concat([raw[["price"]], feats], axis=1)
    keep = (frame["price"].between(*PRICE_BOUNDS)) & (frame["size"].between(*SIZE_BOUNDS))
    frame = frame[keep].copy()
    if verbose:
        print(f"dedup: {n0} -> {n1}  |  bounds: {n1} -> {len(frame)}")
    return frame
