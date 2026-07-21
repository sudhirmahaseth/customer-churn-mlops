FROM python:3.12-slim

WORKDIR /app

# केवल सिस्टम डिपेंडेंसीज इंस्टॉल करना + git (DVC को इसकी जरूरत होती है)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# मुख्य लाइब्रेरीज इंस्टॉल करना (DVC के HTTP सपोर्ट के साथ)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy scikit-learn xgboost mlflow "dvc[http]" fastapi uvicorn

# प्रोजेक्ट कोड कॉपी करना
COPY . /app

# बचे हुए पैकेजेस इंस्टॉल करना
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 🎯 मैजिक स्टेप: बिल्ड के समय डैग्सहब से असली मॉडल पुल करना
ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

# --local फ़्लैग का उपयोग करके क्रेडेंशियल्स को सीधे .dvc/config.local में डाला जाता है 
# इससे सिंटैक्स एरर या स्प्रैडिंग की समस्या नहीं होती और dvc pull फ़ोर्सफुली चलता है
RUN dvc remote modify --local myremote user $DAGSHUB_USERNAME && \
    dvc remote modify --local myremote password $DAGSHUB_TOKEN && \
    dvc pull

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]