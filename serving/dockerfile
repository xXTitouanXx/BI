FROM python:3.10-slim
WORKDIR /app
COPY api.py .
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]