import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load Titanic dataset
@st.cache_data
def load_data():
    titanic = sns.load_dataset('titanic')
    return titanic.copy()

# Prepare the data
def prepare_data(data):
    data['age'].fillna(data['age'].mean(), inplace=True)
    data['fare'].fillna(data['fare'].mean(), inplace=True)
    data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data['embarked'] = data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    data['class'] = data['class'].map({'First': 0, 'Second': 1, 'Third': 2})

    data = data[['sex', 'age', 'fare', 'embarked', 'class', 'survived']]
    data.dropna(inplace=True)

    return data

# Train the model
@st.cache_data
def train_model(data):
    X = data[['sex', 'age', 'fare', 'embarked', 'class']]
    y = data['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, accuracy, X_train, y_train, X_test, y_test, X

# Main app
def app():
    st.title('🛳️ Titanic Survival Predictor')
    st.markdown("Predict the survival of a Titanic passenger based on personal and ticket details ((Model Accuracy: 79.89%)")

    # Sidebar Inputs
    st.sidebar.header("Passenger Info")

    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    age = st.sidebar.slider("Age", 0, 100, 30)
    fare = st.sidebar.slider("Fare (£)", 0, 500, 50)
    embarked = st.sidebar.radio("Embarked From", ["Cherbourg (France)", "Queenstown (Ireland)", "Southampton (England)"])
    pclass = st.sidebar.selectbox("Passenger Class", ["First", "Second", "Third"])
    
    raw_data = load_data()
    prepared = prepare_data(raw_data)
    model, accuracy, X_train, y_train, X_test, y_test, X  = train_model(prepared)
    
    y_pred = model.predict(X_test)
    # Model Evaluation Section
    st.subheader("📈 Model Evaluation")

    # Split into 3 columns
    col1, col2 = st.columns(2)

    with col1:



        st.markdown("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

    with col2:
        st.markdown("**ROC-AUC Score:**")
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        st.metric("ROC-AUC", f"{roc_auc:.2f}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14}  # <- annotation font size
        )

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        ax.tick_params(axis='both', labelsize=11)  # <- tick labels

        st.pyplot(fig)
    with col2:
        # Feature Importance (below the columns)
        st.subheader("🔍 Feature Importance")

        importances = model.feature_importances_
        features = X.columns

        plt.figure(figsize=(8, 6))
        plt.barh(features, importances)
        plt.xlabel("Importance", fontsize=15)
        plt.title("Random Forest Feature Importance", fontsize=8)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()

        st.pyplot(plt)

    # Predict button
    if st.sidebar.button("Predict Survival"):
        sex_num = 0 if sex == "Male" else 1
        pclass_num = {"First": 0, "Second": 1, "Third": 2}[pclass]
        embarked_num = {"Cherbourg (France)": 0, "Queenstown (Ireland)": 1, "Southampton (England)": 2}[embarked]

        input_data = pd.DataFrame({
            'sex': [sex_num],
            'age': [age],
            'fare': [fare],
            'embarked': [embarked_num],
            'class': [pclass_num]
        })

        # Load and prepare data


        # Train and predict
        
        prediction = model.predict(input_data)[0]

        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.success("✅ The passenger likely survived.")
        else:
            st.error("❌ The passenger likely did not survive.")


        




if __name__ == "__main__":
    app()


