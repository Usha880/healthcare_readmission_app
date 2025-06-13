# 🩺 Healthcare Readmission Risk Prediction Dashboard

An interactive Streamlit web app that predicts patient hospital readmission risk using a trained LightGBM model. Built with healthcare data, custom visualizations, and clean UI styling to support data-driven decision making in hospitals.

---

## 🚀 Live Demo

🔗 [Click here to view the app](https://share.streamlit.io/your-streamlit-link)  

---

## 📂 Project Structure
📁 healthcare_readmission-app/
├── app.py # Streamlit app main script
├── lgbm_model1.pkl # Trained LightGBM model
├── requirements.txt # Dependencies for Streamlit Cloud
└── README.md # Project description (this file)

---

## 📊 Features

✅ Upload healthcare patient CSV data  
✅ Visualize:
- Target distribution (readmission)
- Categorical & numerical feature insights
- Correlation heatmaps and boxplots  
✅ Predict readmission risk for selected patient  
✅ Download prediction results  
✅ Fully styled with custom CSS for better UX  

---

## 📁 Dataset Source

This project is based on the **Diabetes 130-US hospitals** dataset:  
🔗 https://www.kaggle.com/datasets/whenamancodes/diabetes-prediction-dataset

---

## ⚙️ Technologies Used

- **Python**
- **Streamlit** – for the web app interface
- **LightGBM** – for high-speed, accurate predictions
- **Pandas & NumPy** – for data manipulation
- **Seaborn & Matplotlib** – for visualizations
- **Custom CSS** – for a clean, responsive design

---

## 🧠 Model Details

The LightGBM model was trained using:
- 48 engineered features (demographics, diagnosis codes, medication)
- Preprocessing steps include missing value imputation and categorical encoding
- The model outputs a binary prediction (Readmitted / Not Readmitted) with probability score

---

## 💡 How to Use

1. Clone this repo or open it on Streamlit Cloud
2. Upload a healthcare patient CSV file
3. Explore the dataset visually
4. Predict readmission risk for any patient row
5. Download the prediction as a CSV

---

## 📌 Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud)  
To deploy your own:
```bash
git clone https://github.com/Usha880/healthcare_readmission-app.git
cd healthcare_readmission-app
streamlit run app.py
