import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'
df = pd.read_csv(url)

# Preprocess the dataset
df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
df = df.dropna()  # Remove rows with missing values

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Define the rules based on feature importances
rules = {}
for i, feature in enumerate(X.columns):
    condition = f"{feature} == 1"
    conclusion = "High risk of cervical cancer" if model.feature_importances_[i] > 0.1 else "Low risk of cervical cancer"
    rules[f"Rule {i+1}"] = {'condition': condition, 'conclusion': conclusion}

# Define the Streamlit app
def main():
    st.title("Cervical Cancer Diagnosis Expert System")
    st.write("Answer the following questions to determine the risk of cervical cancer:")

    # Collect user input
    user_input = {}
    for feature in X.columns:
        if feature in ['First sexual intercourse', 'Num of pregnancies']:
            user_input[feature] = st.number_input(f"Enter your {feature}", min_value=0, step=1, key=feature)
        else:
            user_input[feature] = st.selectbox(f"Select your {feature}", ('0', '1'), key=feature)

    # Apply the rules and make a diagnosis
    diagnosis = apply_rules(user_input)

    # Display the diagnosis
    st.subheader("Diagnosis")
    st.write(diagnosis)

# Apply the rules to make a diagnosis
def apply_rules(user_input):
    for rule_name, rule_data in rules.items():
        condition = rule_data['condition']
        conclusion = rule_data['conclusion']
        if eval(condition, user_input):
            return conclusion

    return "Unable to make a diagnosis"

# Run the Streamlit app
if __name__ == "__main__":
    main()
