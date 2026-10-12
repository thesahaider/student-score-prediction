
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("Student Exam Score Prediction")
st.write(
    "Use machine learning to predict an exam score "
    "based on study hours and previous academic performance."
)

# Load the dataset
df = pd.read_csv("students.csv")

# Select features and target
X = df[["StudyHours", "PreviousScore"]]
y = df["ExamScore"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Training Records", len(X_train))
col2.metric("Mean Absolute Error", f"{mae:.2f} points")
col3.metric("R² Score", f"{r2:.2f}")

st.caption(
    "Evaluation uses a held-out test set. "
    "The dataset is small and fictional, so these metrics "
    "do not establish real-world predictive accuracy."
)

st.divider()

# Prediction inputs
st.subheader("Predict a Student's Exam Score")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Daily study hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5
    )

with col2:
    previous_score = st.slider(
        "Previous score",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )

if st.button("Predict Exam Score", type="primary"):
    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "PreviousScore": [previous_score]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0.0, min(100.0, prediction))

    st.success(f"Predicted exam score: {prediction:.1f} / 100")

# Compare actual and predicted scores
st.subheader("Actual vs. Predicted Scores")

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred
})

st.dataframe(results.round(2), use_container_width=True)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([0, 100], [0, 100], linestyle="--")
ax.set_xlabel("Actual Exam Score")
ax.set_ylabel("Predicted Exam Score")
ax.set_title("Actual vs. Predicted Scores")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.subheader("Training Dataset")
st.dataframe(df, use_container_width=True, hide_index=True)
