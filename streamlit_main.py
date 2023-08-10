import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(layout='wide')
st.title("Student's Performance Predictor Project")
st.header("")

st.sidebar.header("New Student's Data")


def user_def_inputs():
    hours_studied = st.sidebar.slider("Hours Studied", 1, 9, 4)
    previous_score = st.sidebar.slider("Previous Scores", 40, 99, 70)
    sleep_hours = st.sidebar.slider("Sleep Hours", 4, 9, 6)
    papers_practiced = st.sidebar.slider("Sample Question Papers Practiced", 0, 9, 5)
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", ("Yes", "No"))
    data = {"Hours Studied": hours_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": papers_practiced}
    return pd.DataFrame(data, index=[0])


input_df = user_def_inputs()

pipe = joblib.load('pipe')
input_df = pipe.transform(input_df)

spm = joblib.load('students_performance_model')
prediction = spm.predict(input_df)

score = np.round(prediction[0], 2)
if 90 <= score <= 100:
    status = "Outstanding"
    grade = 'O'
elif 80 <= score < 90:
    status = "Excellent"
    grade = 'A+'
elif 70 <= score < 80:
    status = "Very Good"
    grade = 'A'
elif 60 <= score < 70:
    status = "Good"
    grade = 'B+'
elif 55 <= score < 60:
    status = "Above Average"
    grade = 'B'
elif 50 <= score < 55:
    status = "Average"
    grade = 'C'
elif 45 <= score < 50:
    status = "Bad"
    grade = 'C-'
else:
    status = "Need a Lot of Improvement"
    grade = 'D'

score_criteria_df = pd.DataFrame({
    "Performance Range": ["=90 - <=100", "=80 - <90", "=70 - <80", "=60 - <70", "=55 - <60", "=50 - <55", "=45 - <50",
                          "44 and Below"],
    "Grade": ['O', 'A+', 'A', 'B+', 'B', 'C', 'C-', 'D'],
    "Status": ['Outstanding', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Bad',
               'Need a Lot of Improvement']
})

scores_column, score_criteria_column = st.columns((1, 1))

with scores_column:
    st.header("")
    st.header("")
    st.subheader(f"\n\nPerformance Index: **{str(score)}**")
    st.subheader(f"Performance Grade: **{grade}**")
    st.subheader(f"Performance Status: **{status}**")

with score_criteria_column:
    st.write("\n\n **Grading System:** ")
    st.dataframe(score_criteria_df)
