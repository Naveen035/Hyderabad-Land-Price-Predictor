# 🏠 Hyderabad Land Price Predictor

Welcome to the **Hyderabad Land Price Predictor**! This project aims to predict the price of land in Hyderabad based on location and area. It encompasses various stages, from web scraping and data cleaning to data visualization, machine learning, and deployment using Streamlit.

🔗 [GitHub Repository](https://github.com/Naveen035/Hyderabad-Land-Price-Predictor)

---

## 🌟 Project Overview
The **Hyderabad Land Price Predictor** is a data science project that involves:

- 📄 **Web Scraping**: Collected real estate data from [NoBroker](https://www.nobroker.in/blog/property-rates-in-hyderabad/).
- 🛠 **Data Cleaning**: Processed raw data to handle missing values, inconsistencies, and prepared it for analysis.
- 📊 **Data Visualization**: Explored data insights and patterns using detailed plots and charts.
- 🤖 **Machine Learning Prediction**: Built a model to estimate the price per square foot for different areas.
- 🚀 **Deployment**: Deployed the project as a web application using **Streamlit** for an interactive user experience.

---

## 🛠 Technologies Used
- **Programming Language**: Python 🐍
- **Web Scraping**: BeautifulSoup and Requests
- **Data Manipulation**: Pandas
- **Data Visualization**: Matplotlib and Seaborn
- **Machine Learning**: Scikit-learn
- **Web App Deployment**: Streamlit

---

## 🔍 Project Workflow

### 1. **Web Scraping** 🕸️
- **Source**: Data was collected from [NoBroker](https://www.nobroker.in/blog/property-rates-in-hyderabad/) using **BeautifulSoup** and **Requests** libraries.
- **Objective**: Extracted details of land prices and locality information.

### 2. **Data Cleaning** 🧹
- **Data Handling**: Processed the scraped data to remove null values, duplicates, and handled outliers.
- **Encoding**: Used **Label Encoding** for categorical features (e.g., locations).
- **Feature Engineering**: Created additional features as needed for improved model performance.

### 3. **Data Visualization** 📈
- **Tools Used**: Visualizations were created using **Matplotlib** and **Seaborn** to explore relationships between variables.
- **Key Visuals**:
  - Distribution plots for price per square foot.
  - Bar charts comparing prices in various localities.
  - Heatmaps showing correlations between features.

### 4. **Machine Learning Model** 🤖
- **Model Used**: Trained a regression model using **Scikit-learn** to predict the land price based on input features (location and area).
- **Preprocessing**:
  - **Label Encoding** for categorical data.
  - **Feature Scaling** where necessary.
- **Model Evaluation**: Used metrics like **Mean Absolute Error (MAE)** and **R-squared** to validate the model's performance.

### 5. **Deployment** 🚀
- **Platform**: Deployed the app using **Streamlit**, making it accessible through a user-friendly web interface.
- **Interactive Features**:
  - Users can select a location from a dropdown.
  - Input the area in square feet.
  - View predicted prices dynamically.

---

## 📄 How to Run the Project Locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Naveen035/Hyderabad-Land-Price-Predictor.git
   cd Hyderabad-Land-Price-Predictor

---

## 🚀 Future Enhancements
- Integrate additional features like property type, age of property, etc.
- Enhance model accuracy by using advanced ML algorithms or hyperparameter tuning.
- Deploy the app on cloud platforms for wider accessibility.

---

## 👏 Acknowledgements
- Special thanks to [NoBroker](https://www.nobroker.in/blog/property-rates-in-hyderabad/) for providing the data source.
- Inspired by real-world data science and machine learning applications.

---

## ✨ Connect with Me
Feel free to reach out or contribute to this project! 😊

- **LinkedIn**: [Naveen Kumar](https://www.linkedin.com/in/naveen-kumar1002/)
- **GitHub**: [Naveen035](https://github.com/Naveen035)

---

> **Note**: The data and prices used in this project refer to the year 2021. Please be aware that current prices may vary due to market fluctuations and recent developments.
