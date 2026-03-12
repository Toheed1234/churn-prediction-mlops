# Use the official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first (this caches the installed packages to speed up future builds)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of our code into the container
COPY . .

# Expose the port our FastAPI app runs on
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
