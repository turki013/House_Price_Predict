<div align="center">

# 🏠 House Price Predictor
### توقع أسعار العقارات السعودية بالذكاء الاصطناعي

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br>

> نموذج تعلم آلة تفاعلي يتوقع أسعار العقارات السكنية بدقة عالية  
> بناءً على مواصفاتها الإنشائية، مع التركيز على ضبط المعلمات للحصول على أفضل أداء ممكن.

<br>

</div>

---

## ✨ المميزات الرئيسية

| الميزة | الوصف |
|--------|-------|
| 📊 **تحليل ذكي للبيانات** | تصفية البيانات والتركيز على المدخلات الأكثر تأثيراً على السعر |
| 🌲 **خوارزمية Random Forest** | نمذجة العلاقات المعقدة في أسعار العقارات |
| ⚙️ **ضبط دقيق للأداء** | التحكم في عمق الأشجار لتقليل الـ Overfitting |
| 💬 **واجهة تفاعلية (CLI)** | إدخال بيانات العقار واستقبال السعر المتوقع فورياً |

---

## 🛠️ التقنيات المستخدمة

```
Python 3.12   ·   Pandas   ·   Scikit-Learn
```

---

## 📈 مراحل تطوير النموذج

مرّ النموذج بثلاث مراحل للوصول إلى أفضل توازن بين الدقة والاستقرار:

<br>

| # | النموذج | Train Score | Test Score | الملاحظة |
|---|---------|:-----------:|:----------:|----------|
| 1 | `LinearRegression` | 2.8% | 1.9% | ❌ ضعيف — البيانات غير خطية |
| 2 | `RandomForestRegressor` (افتراضي) | 88.0% | 68.0% | ⚠️ Overfitting واضح |
| 3 | `RandomForestRegressor` (مُعدَّل) | 80.4% | **69.5%** | ✅ أداء متوازن ومستقر |

<br>

### ⚙️ إعدادات النموذج النهائي

```python
model = RandomForestRegressor(
    max_depth=5,           # تحديد عمق الشجرة — يمنع الإغراق بتفاصيل البيانات
    min_samples_split=10,  # لا تفرع إلا بوجود 10 عينات مشتركة على الأقل
    random_state=42
)
```

---

## 🚀 التثبيت والتشغيل

**1. استنسخ المستودع**
```bash
git clone https://github.com/turki013/House_Price_Predict.git
cd House_Price_Predict
```

**2. ثبّت المكتبات المطلوبة**
```bash
pip install pandas scikit-learn
```

**3. ضع ملف البيانات في مجلد المشروع**
```
House_Price_Predict/
├── house_price_model.py
└── Aqar_data.csv        ← ضع الملف هنا
```

**4. شغّل النموذج**
```bash
python house_price_model.py
```

---

## 🖥️ مثال على التشغيل

```
────────── تم تدريب النموذج بنجاح ──────────

أدخل مساحة العقار (size)       : 150
أدخل عدد غرف النوم (bedrooms)  : 2
أدخل عدد دورات المياه (bathrooms): 2

────────────────────────────────────────────
  السعر المتوقع للعقار هو: 1,346,747.08 ريال
────────────────────────────────────────────
  تنويه: السعر مجرد توقع — قد يصيب ويخطئ
────────────────────────────────────────────
```

---

## 🗺️ خطط مستقبلية

- [ ] **ترميز الأحياء** — إضافة One-Hot Encoding لأسماء المناطق لزيادة دقة التسعير الجغرافي
- [ ] **واجهة ويب** — بناء تطبيق تفاعلي باستخدام Streamlit
- [ ] **مزيد من الميزات** — إضافة عمر العقار، الطابق، وقرب الخدمات

---

## 🤝 المساهمة

المشروع مفتوح للمساهمين! يسعدنا استقبال Pull Requests وفتح Issues.

```
Fork → Branch → Commit → Pull Request
```

---

## 📄 الترخيص

هذا المشروع مرخص تحت [MIT License](LICENSE).

---

<div align="center">

صُنع بـ 🤍 بواسطة **[turki013](https://github.com/turki013)**

</div>
