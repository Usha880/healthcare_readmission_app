# 🩺 Healthcare Readmission Risk Prediction Dashboard

An interactive Streamlit web app that predicts patient hospital readmission risk using a trained LightGBM model. Built with healthcare data, custom visualizations, and clean UI styling to support data-driven decision making in hospitals.

---

## 🚀 Live Demo

🔗 [Click here to view the app]

(https://healthcare-readmission-app.onrender.com)

---

## 📂 Project Structure

├── app.py 

├── lgbm_model1.pkl 

├── requirements.txt

├── Dockerfile 

└── README.md 


---

## 📊 Features

✅ Upload healthcare patient CSV data  
✅ Visualize:
- Target distribution (readmission)
- Categorical & numerical feature insights
- Correlation heatmaps and boxplots  
✅ Predict readmission risk for selected patient  
✅ Download prediction results  
✅ Fully styled with custom CSS for a smooth user experience  

---

## 📁 Dataset Source

This project uses the **Diabetes 130-US hospitals** dataset:  
🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/whenamancodes/diabetes-prediction-dataset)

---

## ⚙️ Technologies Used

- **Python**
- **Streamlit** – frontend UI framework
- **LightGBM** – machine learning model
- **Pandas & NumPy** – data preprocessing
- **Seaborn & Matplotlib** – visualizations
- **Docker** – containerized deployment on Render
- **Custom CSS** – UI enhancements

---

## 🧠 Model Details

- Trained on 48 key healthcare features (e.g., demographics, diagnosis codes, medications)
- Preprocessing includes:
  - Missing value handling
  - Categorical encoding
- Model Output:
  - Binary prediction: **Readmitted** / **Not Readmitted**
  - Probability score shown with each prediction

---

## 💡 How to Use

1. Open the app using the link above.
2. Upload a healthcare patient dataset in CSV format.
3. Navigate through tabs to explore data.
4. Select a patient record to predict readmission risk.
5. Download the prediction results as a CSV file.

---

## 📌 Deployment

✅ **Deployed on Render**: [https://render.com]

(https://healthcare-readmission-app.onrender.com)  
Used Docker for containerization with the following setup:

### Dockerfile Highlights:
```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]


To Run Locally:

git clone https://github.com/Usha880/healthcare_readmission-app.git
cd healthcare_readmission-app
streamlit run app.py


