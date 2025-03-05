FROM python:3.10-slim
WORKDIR /app
COPY app.py .
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8081", "--server.address=0.0.0.0"]