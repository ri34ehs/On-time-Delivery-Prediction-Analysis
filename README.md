I-Powered Logistics Optimization Project
A Proof-of-Concept for Predicting E-Commerce Shipment Delays
This project is an end-to-end data science solution designed to predict whether an e-commerce shipment will be delivered on time. By analyzing a real-world dataset, this project identifies key drivers of delays and culminates in a deployed, interactive web application that provides real-time risk predictions.

Live Demo: https://rneder-5.onrender.com/! 👈 

🎯 Business Problem
In the competitive e-commerce landscape, late deliveries lead to increased customer service costs, reduced customer loyalty, and operational inefficiencies. This project tackles this problem by building a machine learning model that can proactively identify shipments at a high risk of being late, allowing the business to move from a reactive to a predictive logistics strategy.

✨ Key Features
In-Depth Exploratory Data Analysis (EDA): Uncovered key insights about operational bottlenecks.

Predictive Modeling: Trained and evaluated multiple classification models to achieve a reliable performance.

Actionable Insights: Translated model results into clear, data-driven business recommendations.

Interactive Web App: Deployed the final model as a user-friendly Streamlit application for real-time predictions.

🛠️ Tech Stack
Programming Language: Python

Data Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Web Application: Streamlit

Deployment: Render, GitHub

📂 Repository Structure
├── 📄 app.py                     # The script for the Streamlit web application
├── 📄 late_shipment_predictor.pkl  # The saved, trained machine learning model
├── 📄 requirements.txt           # A list of Python libraries required to run the project
├── 📄 Train.csv                    # The original, raw dataset
├── 📓 Project_Notebook.ipynb       # The main Jupyter Notebook with the full analysis and model training
└── 📄 README.md                    # This file!
🚀 How to Run Locally
To run this project on your own machine, follow these steps:

Clone the repository:

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Install the required libraries:

Bash

pip install -r requirements.txt
Run the Streamlit application:

Bash

streamlit run app.py
Your web browser should open with the application running locally!

📈 Project Workflow & Key Insights
The project followed a structured, four-phase approach:

Phase 1: Exploratory Data Analysis (EDA)
I performed a deep dive into the data, discovering several critical insights:

Shipments sent by 'Ship' were the most likely to be late.

Warehouse F was identified as a major operational bottleneck, handling the most volume and contributing the most delays.

A higher number of customer care calls strongly correlated with late deliveries.

Phase 2: Data Cleaning & Preprocessing
The raw data was cleaned and transformed into a model-ready format using one-hot encoding for categorical features.

Phase 3: Model Building & Evaluation
I trained three models (Logistic Regression, Random Forest, Gradient Boosting). The Logistic Regression model was selected as the champion due to its strong performance and interpretability.

Key Metric Achieved: 70.2% F1-Score on the test set.

Phase 4: Feature Importance & Recommendations
By analyzing the final model, I extracted the following business recommendations:

Target Warehouse A for Improvement: The model identified this warehouse as the primary source of delays.

Optimize Standard Shipments: Low and medium-importance products were the biggest drivers of late deliveries.

Learn from Success: Highly discounted items were delivered faster, suggesting an efficient process that could be replicated.
