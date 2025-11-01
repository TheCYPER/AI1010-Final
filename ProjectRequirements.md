# AI1010 - Introduction to AI  
## Office Category Classification Challenge  
### Data Challenge Assignment - Fall 2025  

---

## 1. Introduction  

Welcome to your first machine learning data challenge! In this project, you will apply the machine learning techniques you’ve learned in class to solve a real-world classification problem.

### 1.1 The Problem  

Imagine you work for a real estate analytics company that needs to automatically categorize office buildings into different quality tiers based on their characteristics. Your task is to build a machine learning model that can predict the OfficeCategory (ranging from 0 to 4, where 0 represents the lowest tier and 4 represents the highest tier) for office buildings based on various features such as:

- Physical characteristics (office space, plot size, number of floors)  
- Quality indicators (building grade, condition)  
- Amenities (parking spots, meeting rooms, restrooms)  
- Location and zoning information  
- And many more features!

This is a multi-class classification problem where you need to predict one of five categories (0, 1, 2, 3, or 4) for each office building.

### 1.2 Competition Details  

This challenge is hosted on Kaggle as an in-class competition. To participate, you must:

1. Create a free Kaggle account if you don’t have one  
2. Register for the competition using this URL:  
   [https://www.kaggle.com/t/ea6f778790f8481bac58d98d25ff4b51](https://www.kaggle.com/t/ea6f778790f8481bac58d98d25ff4b51)

**Important:** You can submit predictions to Kaggle up to **5 times per day**, so use your submissions wisely!

---

## 2. Dataset Description  

### 2.1 Training Data  

- **office_train.csv**: Contains 35,000 office buildings with 79 features and the target variable OfficeCategory  
- The target variable (OfficeCategory) takes values: 0, 1, 2, 3, or 4  

### 2.2 Test Data  

- **office_test.csv**: Contains 15,000 office buildings with the same 79 features but without the OfficeCategory column  
- Your predictions on this dataset determine your Kaggle leaderboard score  

### 2.3 Baseline Template  

- **template.ipynb**: A Jupyter notebook that provides a simple baseline solution  
- This baseline uses Logistic Regression and achieves approximately 52% accuracy  

### 2.4 Understanding the Features  

1. Size Features: OfficeSpace, PlotSize, BasementArea, ParkingArea, etc.  
2. Quality Features: BuildingGrade, BuildingCondition, ExteriorQuality, etc.  
3. Count Features: MeetingRooms, Restrooms, ParkingSpots, TotalRooms, etc.  
4. Year Features: ConstructionYear, RenovationYear  
5. Categorical Features: ZoningClassification, BusinessDistrict, BuildingType, etc.  

*Note: Some features have missing values, which you’ll need to handle appropriately!*

---

## 3. Task and Evaluation  

### 3.1 Your Mission  
Predict the **OfficeCategory (0–4)** for each building in the test set.

### 3.2 Evaluation Metric  

**Accuracy = (Number of Correct Predictions) / (Total Predictions)**  

Example: If you correctly predict 12,000 out of 15,000 buildings,  
`Accuracy = 12000 / 15000 = 0.80 = 80%`

---

## 4. Getting Started  

1. **Set Up Your Environment:** Google Colab (recommended), Jupyter Notebook, or any IDE  
2. **Download Data:** From Kaggle competition page  
3. **Run Baseline:** Execute `template.ipynb`  
4. **Improve Model:** Try new models and techniques  

---

## 5. Useful Python Libraries  

- **pandas**, **numpy** — Data manipulation and computation  
- **matplotlib**, **seaborn** — Visualization  
- **scikit-learn** — ML algorithms  
- **XGBoost**, **LightGBM**, **CatBoost** — Boosting methods  
- **PyTorch**, **TensorFlow/Keras** — Optional deep learning frameworks  

---

## 6. Team Formation and Grading  

### 6.1 Team Requirements  
Teams of **2–3 students**. No solo work allowed.

### 6.2 Grading Breakdown (Total: 50% of module grade)

| Component | Weight |
|------------|--------|
| Oral Presentation | 30% (60% of project grade) |
| Written Report | 10% (20% of project grade) |
| Kaggle Performance & Code | 10% (20% of project grade) |

### 6.3 Kaggle Performance & Code  

Requirements:
- Submit predictions to Kaggle (max 5/day)  
- Code must be reproducible  

Submit a ZIP file (`TeamName_Code.zip`) containing:  
- `code/` folder  
- `README.txt`  
- `requirements.txt`  

---

## 6.4 Written Report  

Max 5 pages (PDF), include team name, student names, and Kaggle usernames.

Sections:
1. **Data Exploration & Preprocessing** (30%)  
2. **Feature Engineering** (35%)  
3. **Model Selection & Tuning** (35%)  

---

## 6.5 Oral Presentation  

- Duration: 15 min + 5 min Q&A  
- Both members must speak  
- Grading: Content (50%), Presentation (30%), Understanding (20%)  

---

## 7. Important Rules & Deadlines  

- Cite all external sources  
- No code plagiarism  
- No external labeled data  
- Kaggle submission format: CSV with columns `Id, OfficeCategory`  
- Moodle submission: one ZIP (`TeamName_Project.zip`)  

---

## 8. Frequently Asked Questions  

Covers:
- External data usage  
- Library permissions  
- Handling overfitting, missing values, categorical encoding  
- Submission format and deadlines  

---

## 9. Resources and Further Reading  

- **Kaggle Learn** (Intro to ML, Feature Engineering)  
- **Fast.ai** Practical Deep Learning  
- **Books:**  
  - *Hands-On Machine Learning* by Aurélien Géron  
  - *ISLR* by James et al.  
  - *Python Data Science Handbook* by VanderPlas  

---

## 10. Final Checklist  

### Code
- Runs cleanly  
- Includes README and requirements.txt  
- Reproducible results  

### Report
- ≤ 5 pages  
- Includes visuals, references  

### Presentation
- Booked slot, both speak  

### Kaggle
- At least one successful submission  

---

## 11. Good Luck!  

> “Start early, experiment often, learn from failures, and have fun!”
