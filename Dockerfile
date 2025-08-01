# ✅ Base image: Python resmi, versi terbaru disarankan
FROM python:3.11-slim

# ✅ Atur direktori kerja di container
WORKDIR /app

# ✅ Salin file ke dalam container
COPY . /app

# ✅ Upgrade pip & install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Expose port FastAPI
EXPOSE 8000

# ✅ Perintah default untuk run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
