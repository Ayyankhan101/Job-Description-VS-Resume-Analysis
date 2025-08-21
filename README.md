# Resume-Job Match Score Dashboard

## 🚀 Overview

This project provides a comprehensive analysis and predictive modeling solution for matching resumes to job descriptions. It includes a Jupyter notebook for exploratory data analysis (EDA) and model building, and a Streamlit dashboard for interactive visualization and real-time prediction. The primary goal is to score the match between a resume and a job description on a scale of 1 to 5.

## ✨ Features

- **Exploratory Data Analysis (EDA):** In-depth analysis of the dataset to understand the distribution of match scores and key skills.
- **Text Preprocessing:** Cleaning and preparing text data (job descriptions and resumes) for modeling.
- **Predictive Modeling:** A Ridge regression model to predict the match score.
- **Interactive Dashboard:** A Streamlit application to visualize the data, analyze skill presence, and predict match scores in real-time.
- **Model Persistence:** The trained model and vectorizer can be saved for future use.

## 📊 Dataset

The project uses the `resume_job_matching_dataset.csv` file, which contains the following columns:

- `job_description`: The text of the job description.
- `resume`: The text of the resume.
- `match_score`: The match score (1-5) between the job description and the resume.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/resume-analysis.git
    cd resume-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Usage

To run the Streamlit dashboard, execute the following command in your terminal:

```bash
streamlit run dashboard.py
```

This will open the dashboard in your web browser, where you can upload your own dataset or use the demo data to explore the analysis and predictions.

## 📂 File Descriptions

- **`Analysis.ipynb`:** A Jupyter notebook containing the detailed exploratory data analysis, text preprocessing, model training, and evaluation.
- **`dashboard.py`:** The Streamlit application that provides an interactive user interface for the project.
- **`resume_job_matching_dataset.csv`:** The dataset used for training and analysis.
- **`requirements.txt`:** A list of the Python libraries required to run the project.

## 🤖 Model

The predictive model is a **Ridge regression** model that uses **TF-IDF (Term Frequency-Inverse Document Frequency)** features extracted from the combined text of the job description and resume. The model is trained to predict the `match_score`.

- **MAE (Mean Absolute Error):** ~0.78
- **R² (R-squared):** ~0.33

## 📈 Visualizations

The dashboard includes several visualizations to help you understand the data and the model's performance:

- **Distribution of Match Scores:** A bar chart showing the frequency of each match score.
- **Skill Presence Radar:** A radar chart comparing the frequency of key skills in low-scoring and high-scoring job descriptions.
- **Word Cloud:** A word cloud of the most common skills in high-scoring job descriptions.
- **Residuals vs. Predicted:** A scatter plot to diagnose the model's prediction errors.
- **True vs. Predicted:** A scatter plot to assess the model's predictive accuracy.
