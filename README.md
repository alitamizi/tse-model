# tse-model

مدل‌ها، قاعده‌ها، سیگنال‌ها و حکم‌های Claude برای بورس تهران. داده‌های خام از `alitamizi/tse-data` خوانده می‌شود.

- `state/daily_1399plus.csv.xz` — دادهٔ روزانهٔ تعدیل‌شدهٔ همهٔ نمادها از ۱۳۹۹ (از فایل‌های پویا)
- `state/daily/YYYY-MM-DD.csv` — ردیف هر جلسهٔ جدید که از دادهٔ زنده اضافه می‌شود
- `state/signals.csv` — همهٔ هشدارها و نامزدهای مدل، با زمان و قیمت
- `state/verdicts.csv` — حکم‌های Claude (تأیید/رد) با ورود، هدف و حد ضرر
- `state/evaluation.csv` — نتیجهٔ سیگنال‌ها (درست/غلط/باز)
- `models/five_models.pkl` — مدل ۵٪ (خرید و فروش، ۱۰ روز معاملاتی)
- `code/` — کدها (`live.py intraday` و `live.py close`)
