<div align="center">
  <img src="assets/logo.png" alt="Churn Prediction Logo" width="200" />
</div>

# Customer Churn Prediction: MLOps Pipeline

This is an end-to-end Machine Learning pipeline designed to predict customer churn. The project demonstrates model training alongside software engineering and MLOps best practices, including containerization, automated testing, and CI/CD.

## Tech Stack
* **Machine Learning:** Scikit-learn (Random Forest), Pandas
* **Model Serving:** FastAPI, Uvicorn, Pydantic
* **Frontend Dashboard:** Streamlit
* **Automated Testing:** Pytest
* **Containerization:** Docker
* **CI/CD:** GitHub Actions

## Architecture

1. **`train.py`:** Downloads the IBM Telco Customer Churn dataset, cleans the data, trains a Random Forest Classifier, and exports the model artifact.
2. **`app.py`:** A FastAPI web server that loads the trained model into memory and exposes a `/predict` endpoint for inference.
3. **`frontend.py`:** A Streamlit web application providing an interactive user interface for business stakeholders to input customer features and view churn probabilities.
4. **`test_app.py`:** A Pytest suite verifying the API endpoints and prediction logic.
5. **`Dockerfile`:** Containerizes the API for consistent deployment across environments.
6. **`.github/workflows/test.yml`:** Automates testing on every push to the main branch.

---

## How to Run Locally

### Option 1: Docker (Recommended)
You only need Docker installed on your system to run the API.

```bash
# 1. Build the Docker Image
docker build -t churn-api .

# 2. Run the Container
docker run -p 8000:8000 churn-api
```
Once the application starts, navigate to `http://localhost:8000/docs` to view the interactive API documentation.

### Option 2: Python Virtual Environment
For local development, you can set up a Python virtual environment.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate  

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Generate the Machine Learning Model
python train.py

# 4. Start the API Server
uvicorn app:app --reload
```

---

## Interactive Dashboard (Streamlit)
To use the graphical interface, ensure your virtual environment is activated and the dependencies are installed.

Open a terminal and run:
```bash
streamlit run frontend.py
```
This will launch the web dashboard at `http://localhost:8501`.

---

## API Usage

You can test the API by sending a POST request to the `/predict` endpoint.

**Example Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 1,
  "MonthlyCharges": 99.00,
  "TotalCharges": 99.00,
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month"
}
```

**Example Response:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.82
}
```
