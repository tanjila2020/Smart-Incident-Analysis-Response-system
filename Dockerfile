# Dockerfile -- containerizes the Smart Incident FastAPI service
# ---------------------------------------------------------------
# Packages the app + dependencies + Python runtime into one image that runs
# identically on any machine.
#
# Build:  docker build -t incident-api .
# Run:    docker run -p 8000:8000 incident-api
# Visit:  http://localhost:8000/docs

FROM python:3.11-slim
WORKDIR /app

# copy deps first so Docker caches the install layer (faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the app code and the trained model
COPY app.py .
COPY incident_model.pkl .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
