FROM python:3.12-alpine

# Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# Copia il requirements.txt e installa le dipendenze
COPY requirements.txt /app/

RUN apk update && apk add --no-cache \
    build-base \
    gfortran \
    python3-dev \
    musl-dev

RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file di codice nell'immagine
COPY . /app

# Comando per avviare l'applicazione Streamlit
CMD ["python", "mqtt/MosquittoSubscriber.py"]

