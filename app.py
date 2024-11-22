import streamlit as st
import pandas as pd
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


st.write("""
	# Lung Cancer Mortality Prediction App

	This app predicts the lung cancer mortality using SVM

	""")

st.sidebar.header('User Input Features')

def user_input_features():
	gender = st.sidebar.selectbox('Sex', ('Male', 'Female'), index=None)
	cancer_stage = st.sidebar.selectbox('Cancer Stage', ('Stage I', 'Stage II', 'Stage III', 'Stage IV'), index=None)
	family_history = st.sidebar.selectbox('Family History', ('Yes', 'No'), index=None)
	smoking_status = st.sidebar.selectbox('Smoking Status', ('Never Smoked','Former Smoker', 'Current Smoker', 'Passive Smoker'), index=None)
	hypertension = st.sidebar.selectbox('Hypertension', ('Yes', 'No'), index=None)
	asthma = st.sidebar.selectbox('Asthma', ('Yes', 'No'), index=None)
	cirrhosis = st.sidebar.selectbox('Cirrhosis', ('Yes', 'No'), index=None)
	treatment_type = st.sidebar.selectbox('Treatment Type', ('Surgery','Chemotherapy', 'Radiation', 'Combined'), index=None)

	data = {
	'gender': gender,
	'cancer_stage': cancer_stage,
	'family_history': family_history, 
	'smoking_status': smoking_status, 
	'hypertension': hypertension, 
	'asthma': asthma, 
	'cirrhosis': cirrhosis,
	'treatment_type': treatment_type 
	}
	features = pd.DataFrame(data, index=[0])
	return features

input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# The sole purpose of combining the entire dataset is for encoding purposes. So it gets standardized & it is easier, less mistake
penguins_raw = pd.read_csv('data/lung_cancer_mortality_cleaned.csv')
penguins = penguins_raw.drop(columns=['survived'])
df = pd.concat([input_df, penguins], axis=0) 
#axis 0 means it is concated horizontally. axis 1 is concated vertically
# st.write(df)

# Encoding of ordinal features
encode = ['gender', 'cancer_stage', 'family_history', 'smoking_status', 'hypertension', 
				'asthma', 'cirrhosis', 'treatment_type']
for col in encode:
	#get_dummies returns a dataframe with encoded values. df[col] is the data, prefix=col is the column name
	dummy = pd.get_dummies(df[col], prefix=col)
	df = pd.concat([df,dummy],axis=1)
	del df[col]
df = df[:1] #Selects only the first row (user input data) -> THIS. Bcs you don't need the whole dataset.
# st.write(df)

#Display user input features
st.subheader('User Input Feature')

st.write('Using input parameters (shown below)')
st.write(df)


# Pickle file for machine learning classifier training
# https://www.datacamp.com/tutorial/pickle-python-tutorial
#Load a saved classification model. This part, we need a pickle file (executed & generated in another python file)
load_clf = pickle.load(open('survival_svm_clf.pkl', 'rb'))

#Apply model for our specific input for predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf._predict_proba_lr(df)

st.subheader('Prediction')
survival = np.array(['No', 'Yes'])
st.write(survival[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


st.sidebar.markdown("""
	[Go back to portfolio](ainurafifah00.github.io)
	""")



