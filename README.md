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

مرّ النموذج بعدة مراحل للوصول إلى أفضل توازن بين دقة التوقع والتعميم لبيانات جديدة:

<br>

| # | النموذج والتقنيات المستخدمة | Train Score | Test Score | الملاحظة |
|---|-----------------------------|:-----------:|:----------:|----------|
| 1 | `RandomForestRegressor` (افتراضي مع OHE) | 95.1% | 64.3% | ❌ Overfitting شديد فجوة ~31% |
| 2 | `RandomForestRegressor` + Smooth Encoding | 89.7% | 65.8% | ⚠️ تسريب بيانات جزئي واستجابة خطية معقدة |
| 3 | **`RandomForestRegressor` + Log Transform + Smooth Encoding** | **82.12%** | **76.57%** | ✅ **أداء ممتاز ومستقر (فجوة 4% فقط)** |

<br>

### ⚙️ إعدادات النموذج النهائي

```python
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,           # تحديد عمق الشجرة يمنع حفظ البيانات
    min_samples_leaf=10,   # اشتراط وجود 10 عقارات على الأقل في كل ورقة
    max_features=0.8,      # اختيار عشوائي لـ 80% من الميزات لتقليل الـ Variance
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

## 🖥️ مثال على التشغيل

```text
────────── تم تدريب النموذج بنجاح ──────────

أدخل مساحة العقار من فضلك (رقم فقط) : 250
أدخل عدد غرف النوم (رقم فقط)     : 4
أدخل عدد دورات المياه (رقم فقط)   : 3
أدخل اسم الحي                     : النرجس
نوع العقار (فيلا / شقة / غير ذلك): عقار
هل يوجد غرفة سائق؟ (1 نعم / 0 لا): 0
هل يوجد غرفة خادمة؟ (1 نعم / 0 لا): 0
هل العقار جديد؟ (1 نعم / 0 لا): 1
────────────────────────────────────────────
  السعر المتوقع للعقار هو: 1,423,718.83 ريال
────────────────────────────────────────────
  تنويه: السعر مجرد توقع من النموذج — قد يصيب ويخطئ
────────────────────────────────────────────
```

---

## 🗺️ خطط مستقبلية

- [✓] **ترميز الأحياء** — إضافة One-Hot Encoding لأسماء المناطق لزيادة دقة التسعير الجغرافي
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
