FROM python:3.12-slim

WORKDIR /app

# केवल सिस्टम डिपेंडेंसीज इंस्टॉल करना
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# मुख्य लाइब्रेरीज इंस्टॉल करना
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy scikit-learn xgboost mlflow dvc fastapi uvicorn

# प्रोजेक्ट कोड कॉपी करना
COPY . /app

# बचे हुए पैकेजेस इंस्टॉल करना
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8000

# ✅ यहाँ हाइफ़न (--) को ठीक कर दिया गया है
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]