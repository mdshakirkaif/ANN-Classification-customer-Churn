import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# Load the trained model
model=tf.keras.models.load_model('model.h5')

with open('labelencoder.pkl','rb') as file:
    label_encoder=pickle.load(file)
    
with open('onehotencoder.pkl','rb') as file:
    one_hotencoder=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')

# User Inputs
Geography=st.selectbox('Geography',one_hotencoder.categories_[0])
Gender=st.selectbox('Gender',label_encoder.classes_)
Age=st.slider("Age",18,92)
Balance=st.number_input("Balance")
CreditScore=st.number_input("CreditScore")
EstimatedSalary=st.number_input("EstimatedSalary")
Tenure=st.slider('Tenure',0,10)
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('Has Credit card',[0,1])
IsActiveMember=st.selectbox('Is active member',[0,1])

#Prepare the input data 

# input data
input_data=pd.DataFrame({
    'CreditScore':[CreditScore],
    'Gender':[label_encoder.transform([Gender])[0]],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
})

# One-Hot-Encoding
onehotencoded=one_hotencoder.transform([[Geography]]).toarray()
encoded_df=pd.DataFrame(onehotencoded,columns=one_hotencoder.get_feature_names_out())

# Combine one hot columns with input data 
input_df=pd.concat([input_data.reset_index(drop=True),encoded_df],axis=1)

# scale the input data
input_data_scaled=scaler.transform(input_df)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

if prediction_prob>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')