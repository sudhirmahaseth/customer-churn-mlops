FROM python:3.12-slim

WORKDIR /app

# केवल सिस्टम डिपेंडेंसीज इंस्टॉल करना (यदि आवश्यक हो)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 🚀 पहले मुख्य लाइब्रेरीज को बिना कड़े वर्जन्स के सीधे इंस्टॉल करना (ताकि रेजोल्यूशन फ़ास्ट हो)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy scikit-learn xgboost mlflow dvc fastapi uvicorn

# प्रोजेक्ट कोड कॉपी करना
COPY . /app

# यदि requirements.txt में कोई अन्य पैकेज बचे हैं तो उन्हें इंस्टॉल करना
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "---host", "0.0.0.0", "---port", "8000"]