# Diabetes Prediction using Logistic Regression

**Midterm Project for ML Zoomcamp 2025**

## 📋 Description

This project focuses on predicting diabetes based on patient health data. Using logistic regression, the model predicts the probability of a patient having diabetes. A Flask API is included to serve the model and make real-time predictions for new patients.

## 🗂️ Repository Structure

```
.
├── train.py                      # Script to train the model
├── predict.py                    # Flask API for predictions
├── predict_test.py               # Script to test the API
├── model_C=1.0_diabetes.bin      # Trained model file
├── Dockerfile                    # Docker configuration
├── Pipfile                       # Pipenv dependencies
├── Pipfile.lock                  # Locked Pipenv dependencies
├── notebook.ipynb                # Diabetes Prediction Model Development
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Pipenv (for dependency management)
- Docker (optional, for containerized deployment)

### 1. Install Dependencies

Install Pipenv if you haven't already:

```bash
pip install pipenv
```

Install project dependencies:

```bash
pipenv install --dev
```

Activate the virtual environment:

```bash
pipenv shell
```

### 2. Train the Model

Run the training script to train a new model:

```bash
python train.py
```

The trained model will be saved as `model_C=1.0_diabetes.bin`.

### 3. Run the Flask API

Start the API server locally:

```bash
python predict.py
```

The API will be available at:
```
http://localhost:9696/predict
```

### 4. Test the API

Test the API using the included test script:

```bash
python predict_test.py
```

Example request:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": value1, "feature2": value2, ...}'
```

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t diabetes-api .
```

### Run the Container

```bash
docker run -it --rm -p 9696:9696 diabetes-api
```

The API will be accessible at `http://localhost:9696/predict`.

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Regularization Parameter**: C = 1.0
- **Classification Threshold**: 0.38 (optimized for precision ≈ recall)

## 🔍 API Response Format

The API returns a JSON object with the following structure:

```json
{
  "diabetes_probability": 0.45,
  "diabetes": false
}
```

- `diabetes_probability`: Predicted probability of diabetes (0.0 to 1.0)
- `diabetes`: Boolean prediction (true/false)

## 📝 Notes

- The default threshold for classifying diabetes is **0.38**, where precision ≈ recall
- Adjust the threshold in `predict.py` based on your use case requirements
- For production deployment, consider using a production-grade WSGI server like Gunicorn

## 🤝 Contributing

This is a course project for ML Zoomcamp 2025. Feedback and suggestions are welcome!

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**ML Zoomcamp 2025** | Midterm Project
