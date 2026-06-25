# 1. बेस इमेज का चयन (Python 3.12 स्लिम वर्जन - ताकि इमेज का साइज छोटा रहे)
FROM python:3.12-slim

# 2. सिस्टम डिपेंडेंसीज इंस्टॉल करें (यदि कुछ पैकेजेस को कंपाइल करने की जरूरत पड़े)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. कंटेनर के अंदर वर्किंग डायरेक्टरी सेट करें
WORKDIR /app

# 4. डिपेंडेंसीज को कंटेनर में कॉपी करें और इंस्टॉल करें
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. प्रोजेक्ट के सभी जरूरी फोल्डर्स और फाइल्स को कॉपी करें
COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY models/ ./models/

# 6. पोर्ट 8000 को एक्सपोज करें (FastAPI इसी पोर्ट पर चलेगा)
EXPOSE 8000

# 7. कंटेनर स्टार्ट होते ही FastAPI सर्वर को रन करने की कमांड
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]