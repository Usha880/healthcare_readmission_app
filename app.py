import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#Page config
st.set_page_config(page_title="Healthcare Readmission Dashboard", layout="wide")

#Load Model
@st.cache_resource
def load_model():
    with open("lgbm_model1.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

feature_names = [
    'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty',
    'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'metformin-glipizide'  
]

#Preprocessing Function
def preprocess_input(input_df):
    df = input_df.copy()
    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
    df.fillna("Missing", inplace=True)
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

def preprocess_input(input_df):
    df = input_df.copy()
    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col]).astype('int64') 
    df.fillna("Missing", inplace=True)
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df

#CSS
st.markdown("""
<style>
/* App Background */
.stApp {
    background-color: #fff9f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffe8dc !important;
    padding: 1.5rem !important;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Text & Headings */
h1, h2, h3, h4, h5, h6,
label, span, p, div,
.stMarkdown, .stText, .stAlert,
.stSuccess, .stWarning, .stInfo,
[data-testid="stMarkdownContainer"] * {
    color: #000000 !important;
}

/* Tables */
thead, tbody, th, td {
    color: #000000 !important;
}
.stDataFrame, .stTable {
    background-color: white !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

/* Radio Buttons */
.stRadio label {
    font-weight: 600 !important;
    color: #000000 !important;
}
.stRadio div[role="radiogroup"] > label {
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    background-color: transparent !important;
    transition: background-color 0.2s ease;
}
.stRadio div[role="radiogroup"] > label:hover {
    background-color: rgba(161, 0, 0, 0.1) !important;
}
.stRadio div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(161, 0, 0, 0.2) !important;
    font-weight: 700 !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: white !important;
    color: #000 !important;
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
}

/* File uploader */
.stFileUploader {
    background-color: white !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}
.stFileUploader label, .stFileUploader span {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background-color: #a10000 !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #d42b2b !important;
    color: white !important;
}

/*Download button main text */
.stDownloadButton button {
    color: white !important;
}

/*Download helper text */
section[data-testid="stDownloadButtonContainer"] label {
    color: white !important;
    font-weight: 500 !important;
}

/* Dropdown - selected value text and container */
div[data-baseweb="select"] > div {
    background-color: white !important; /* White background for the dropdown container */
    color: #000000 !important; /* Black text for the selected value */
    border-radius: 6px !important;
    border: 1px solid #ccc !important; /* Add a border for better visibility */
}

/* Dropdown - selected value text (ensure override) */
div[data-baseweb="select"] div[aria-selected="true"] {
    color: #000000 !important; /* Black text */
}

/* Dropdown - menu item text and background */
div[data-baseweb="select"] div[role="option"] {
    color: #000000 !important; /* Black text for dropdown options */
    background-color: white !important; /* White background for dropdown options */
}
div[data-baseweb="select"] div[role="option"]:hover {
    background-color: #f0f0f0 !important; /* Light gray hover effect for contrast */
}

/* Dropdown - input search bar (if any) */
div[data-baseweb="select"] input {
    color: #000000 !important; /* Black text for search bar */
}

/* Dropdown - menu container (the dropdown list background) */
div[data-baseweb="select"] ul {
    background-color: white !important; /* White background for the dropdown menu */
    border: 1px solid #ccc !important; /* Add a border for better visibility */
}

/* Tabs */
.stTabs [data-baseweb="select"] button {
    color: #000000 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="select"] button[aria-selected="true"] {
    color: #a10000 !important;
    border-bottom: 2px solid #a10000 !important;
}

/* Sliders */
.stSlider .thumb {
    background-color: #a10000 !important;
}
.stSlider .track {
    background-color: #d42b2b !important;
}

/* Checkboxes */
.stCheckbox label {
    color: #000000 !important;
    font-weight: 500 !important;
}

/* Expanders */
.stExpander .stExpanderHeader {
    color: #000000 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


#Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Explore Data", "Predict Readmission"])

#Upload Page
if page == "Upload Data":
    st.title("📤 Upload Healthcare CSV File")
    st.markdown("Upload your dataset in CSV format to begin analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="visible")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.success("**File uploaded successfully!**")  
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

#Explore Page
elif page == "Explore Data":
    st.title("🔍 Explore Dataset")

    if "data" in st.session_state:
        df = st.session_state["data"]

        st.subheader("Basic Information")
        
        #Data types in one box
        with st.container():
            st.markdown("**Dataset Overview**")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            st.write("**Data types:**")
            st.write(df.dtypes)
        
        # Missing values in separate box below
        with st.container():
            st.markdown("**Missing Values**")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                missing_data[missing_data > 0].plot(kind='bar', color='#a10000', ax=ax)
                ax.set_title('Missing Values Count', fontweight='bold')
                ax.set_ylabel('Count')
                st.pyplot(fig)
            else:
                st.write("No missing values found in the dataset.")

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        # New Visualizations Section
        st.subheader("Advanced Visualizations")
        
        # Tab layout for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Numerical Features", "Categorical Features", "Time Series"])
        
        with tab1:
            if "readmitted" in df.columns:
                st.markdown("**Target Variable Analysis**")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=df, x="readmitted", ax=ax, palette="Reds")
                    ax.set_title("Readmission Count", fontweight='bold')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df['readmitted'].value_counts().plot.pie(autopct='%1.1f%%', 
                                                          colors=['#ff9999','#ff6961'], 
                                                          ax=ax)
                    ax.set_title("Readmission Proportion", fontweight='bold')
                    st.pyplot(fig)
            else:
                st.warning("No 'readmitted' column found for target analysis")
        
        with tab2:
            st.markdown("**Numerical Features Analysis**")
            numeric_cols = df.select_dtypes(include=np.number).columns
            
            if len(numeric_cols) > 0:
                selected_num_col = st.selectbox("Select numerical column", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df[selected_num_col], kde=True, color='#a10000', ax=ax)
                    ax.set_title(f"Distribution of {selected_num_col}", fontweight='bold')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x=df[selected_num_col], color='#a10000', ax=ax)
                    ax.set_title(f"Boxplot of {selected_num_col}", fontweight='bold')
                    st.pyplot(fig)
                
                # Correlation heatmap for numerical columns
                if len(numeric_cols) > 1:
                    st.markdown("**Correlation Heatmap**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', annot=True, 
                                fmt=".2f", linewidths=.5, ax=ax)
                    ax.set_title("Numerical Features Correlation", fontweight='bold')
                    st.pyplot(fig)
            else:
                st.warning("No numerical columns found in dataset")
        
        with tab3:
            st.markdown("**Categorical Features Analysis**")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                selected_cat_col = st.selectbox("Select categorical column", cat_cols)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, y=selected_cat_col, order=df[selected_cat_col].value_counts().index, 
                             palette="Reds_r", ax=ax)
                ax.set_title(f"Distribution of {selected_cat_col}", fontweight='bold')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Cross tab with target if available
                if "readmitted" in df.columns:
                    st.markdown("**Relationship with Readmission**")
                    cross_tab = pd.crosstab(df[selected_cat_col], df['readmitted'], normalize='index') * 100
                    cross_tab = cross_tab.sort_values(cross_tab.columns[0], ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    cross_tab.plot(kind='bar', stacked=True, color=['#a10000','#ff9999'], ax=ax)
                    ax.set_title(f"Readmission Rate by {selected_cat_col}", fontweight='bold')
                    ax.set_ylabel("Percentage")
                    ax.legend(title='Readmitted', bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)
            else:
                st.warning("No categorical columns found in dataset")
        
        with tab4:
            st.markdown("**Time-based Analysis**")
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if len(date_cols) > 0:
                selected_date_col = st.selectbox("Select date/time column", date_cols)
                
                try:
                    df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                    time_series = df[selected_date_col].dt.to_period('M').value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    time_series.plot(kind='line', marker='o', color='#a10000', ax=ax)
                    ax.set_title(f"Trend Over Time ({selected_date_col})", fontweight='bold')
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                except:
                    st.warning("Could not convert this column to datetime format")
            else:
                st.warning("No date/time columns found in dataset")
    else:
        st.warning("Please upload a dataset first from the 'Upload Data' page.")

# Predict Page
elif page == "Predict Readmission":
    st.title("🩺 Predict Readmission Risk")
    st.markdown("Use this section to predict patient readmission risk based on the uploaded data.")

    if "data" in st.session_state:
        df = st.session_state["data"]

        st.subheader("Select Patient Record")
        idx = st.number_input("Enter row index", min_value=0, max_value=len(df)-1, value=0, step=1)
        input_row = df.iloc[[idx]]

        st.markdown("**Selected Patient Data**")
        st.dataframe(input_row)

        if st.button("Predict Readmission", type="primary"):
            with st.spinner("Making prediction..."):
                processed_input = preprocess_input(input_row.drop(columns=["readmitted"], errors='ignore'))
                prediction = model.predict(processed_input)
                probability = model.predict_proba(processed_input)[0][1]
                
                if prediction[0] == 1:
                    st.error(f"**High Risk**: This patient has a {probability*100:.1f}% chance of readmission")
                else:
                    st.success(f"**Low Risk**: This patient has a {probability*100:.1f}% chance of readmission")

                # Export prediction
                pred_df = input_row.copy()
                pred_df["Prediction"] = "Readmitted" if prediction[0] == 1 else "Not Readmitted"
                pred_df["Probability"] = probability
                csv = pred_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Prediction Result",
                    data=csv,
                    file_name="prediction_result.csv",
                    mime="text/csv",
                    help="Click to download the prediction results"
                )
    else:
        st.warning("Please upload a dataset first from the 'Upload Data' page.")