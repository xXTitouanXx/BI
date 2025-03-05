FROM python:3.10-slim
WORKDIR /app
COPY project.py .
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["sh", "-c", "python project.py && evidently ui --host=0.0.0.0 --port=8082"]
