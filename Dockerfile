# 1. Base Image
# Use a lightweight Python environment
FROM python:3.9-slim

# 2. Set Working Directory inside the container
WORKDIR /app

# 3. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Source Code
# This copies your 'src' folder (which contains app.py and the saved model.joblib)
# into the container.
COPY src/ src/

# 5. Expose the Port
# The API listens on port 8000
EXPOSE 8000

# 6. Run the Service
# We change directory to 'src' so the app can find the model.joblib easily
WORKDIR /app/src
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]