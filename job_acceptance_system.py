import pandas as pd
import numpy as np
import pymysql
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             classification_report, confusion_matrix)


df = pd.read_csv(r"C:\Data science and gen AI\Practice\project_3\HR_Job_Placement_Dataset.csv")
df.head()

df.shape #(51500, 26)
round(df.isnull().mean()*100,2)

num_col = df.select_dtypes(include=['int64', 'float64']).columns
num_col # 14

cat_col = df.select_dtypes(include=['str']).columns
cat_col # 12

# Data cleaning 

df_1 = df.copy()

# Numerical col with null value:

# The columns which comes under the 20% category used this median imputation method (numerical columns)
numerical = ["ssc_percentage","hsc_percentage","employment_gap_months","notice_period_days"]
for col in numerical:
    df_1[col] = df_1[col].fillna(df_1[col].median())

# Categorical col with null value

# The columns which comes under the 20% category used this mode imputation method (categorical columns)
categorical = ["career_switch_willingness","relevant_experience","job_role_match","layoff_history","relocation_willingness"]
for col in categorical:
    df_1[col] = df_1[col].fillna(df_1[col].mode()[0])

round(df_1.isnull().mean()*100,2)

for c in cat_col:
    df_1[c] = df_1[c].str.strip().str.title()

df_1.head()

df_1["status_flag"] = (df_1["status"] == "Placed").astype(int)

print("before:", df_1.shape)
print("duplicates:", df_1.duplicated().sum())
df_1 = df_1.drop_duplicates().reset_index(drop=True)
print("after:", df_1.shape)

# Feature Engineering

# Experience category flagging based on years_of_experience
def Experience_category(year):
    if year <= 1:
        return "Fresher"
    elif year <= 3:
        return "Junior"
    else:
        return "Senior"

df_1["Experience_category"] = df_1["years_of_experience"].apply(Experience_category)

# Academic performance band based on degree_percentage
def academic_band(score):
    if score < 65:
        return "Average"
    elif score < 75:
        return "Good"
    else:
        return "Excellent"

df_1["Academic_band"] = df_1["degree_percentage"].apply(academic_band)

# Skills match level based on skills_match_percentage
def skills_match(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"

df_1["Skills_match"] = df_1["skills_match_percentage"].apply(skills_match)

# Interview performance category based on average of the three scores
df_1["interview_avg"] = df_1[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1)
def interview_category(score):
    if score < 60:
        return "Developing"
    elif score < 70:
        return "Proficient"
    else:
        return "Outstanding"

df_1["Interview_performance"] = df_1["interview_avg"].apply(interview_category)

# Database Creation and Data Insertion [SQL]
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="makesh28"
)

cursor = connection.cursor()

cursor.execute("CREATE DATABASE IF NOT EXISTS job_acceptance_db")

cursor.close()
connection.close()

print("Database created successfully")

engine = create_engine(
    f"mysql+pymysql://root:makesh28@localhost:3306/job_acceptance_db"
)

df_1.to_sql(
    "candidates",
    con=engine,
    if_exists="replace",
    index=False
)
print("Table created and data inserted successfully")

#Creating a CSV file for the cleaned dataset
df_1.to_csv("Cleaned_Dataset.csv", index=False)

# Model Training 

df = pd.read_csv("Cleaned_Dataset.csv")
df.shape 

target = "status_flag"
drop_cols = ["status", "status_flag", "interview_avg",
             "Experience_category", "Academic_band",
             "Skills_match", "Interview_performance"]

X = df.drop(columns=drop_cols)
y = df[target]
X.shape

binary_cols = ["gender", "internship_experience", "career_switch_willingness",
               "relevant_experience", "job_role_match", "bond_requirement",
               "layoff_history", "relocation_willingness"]
ordinal_cols = ["company_tier", "competition_level"]
onehot_cols = ["degree_specialization"]
numeric_cols = [c for c in X.columns
                if c not in binary_cols + ordinal_cols + onehot_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.mean().round(4), y_test.mean().round(4)) 

preprocessor = ColumnTransformer(transformers=[
    ("bin", OrdinalEncoder(),binary_cols),

    ("ord", OrdinalEncoder(categories=[["Tier 1", "Tier 2", "Tier 3"],
                                       ["Low", "Medium", "High"]]),
            ordinal_cols),

    ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),

    ("num", StandardScaler(), numeric_cols),
])

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

print(X_train_t.shape, X_test_t.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=["accuracy", "f1", "roc_auc"])
    results.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
    })

cv_results = pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False).round(4)
cv_results

gb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42)),
])

gb_pipe.fit(X_train, y_train)

y_pred = gb_pipe.predict(X_test)                 
y_prob = gb_pipe.predict_proba(X_test)[:, 1]    

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

oof_prob = cross_val_predict(gb_pipe,X_train,y_train,cv=cv,method="predict_proba",n_jobs=-1)[:, 1]

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

results = []

for threshold in thresholds:

    y_pred_threshold = (oof_prob >= threshold).astype(int)

    results.append({
        'Threshold': threshold,
        'Precision': precision_score(y_train, y_pred_threshold),
        'Recall': recall_score(y_train, y_pred_threshold),
        'F1 Score': f1_score(y_train, y_pred_threshold)
    })

threshold_df = pd.DataFrame(results)

threshold_df.round(4)

threshold = 0.40

y_pred_final = (y_prob >= threshold).astype(int)
print("Accuracy :", round(accuracy_score(y_test, y_pred_final), 4))
print("Precision:", round(precision_score(y_test, y_pred_final), 4))
print("Recall   :", round(recall_score(y_test, y_pred_final), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_final), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

model = {
    "pipeline": gb_pipe,
    "threshold": threshold
}
joblib.dump(model, "job_acceptance_model.pkl")
