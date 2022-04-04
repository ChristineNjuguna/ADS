import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("banking_churn.csv")


df =df.replace({'IsActiveMember': {0: "No", 1: "Yes"}})
df =df.replace({'HasCrCard': {0: "No", 1: "Yes"}})
#df['IsActiveMember','HasCrCard'] = df.loc[:,['IsActiveMember','HasCrCard']].replace([0, 1],["No", "Yes"])

# Title
st.header("Bank Churn Analysis.")

col1, col2 = st.columns((1,15))

img = Image.open("arrow-left.png")
col1.image(img,width=50)

col2.subheader("Use the Calculator on the left to predict Churn.")

clist = df['Geography'].unique()
country = st.selectbox("Select a country:",clist)


fig = px.histogram(df[df['Geography'] == country], 
    x = "Gender", y = "Exited", title = f"Churned Clients in {country} grouped by Gender")
st.plotly_chart(fig)

fig2 = px.histogram(df[df['Geography'] == country], 
    x = "IsActiveMember", y = "Exited", title = f"Churned Clients in {country} grouped by Active Status")
st.plotly_chart(fig2)

fig3 = px.histogram(df[df['Geography'] == country], 
    x = "HasCrCard", y = "Exited", title = f"Churned Clients in {country} grouped by Credit Card Status")
st.plotly_chart(fig3)

with st.sidebar:

    # Title
    st.header("Churn Calculator")

    # Input bar 1
    CreditScore = st.number_input("Enter Clients Credit Score", min_value=250, max_value=900,value=600
    ,format="%i")

    # Input bar 1
    Balance = st.number_input("Enter Clients Balance")

    # Input bar 1
    EstimatedSalary = st.number_input("Enter Clients Estimated Salary")



    # Dropdown input
    Geography = st.selectbox("Select their Country", ("France", "Spain","Germany"))

    # Dropdown input
    Gender = st.selectbox("Select their Gender", ("Female", "Male"))

    #Tenure = st.selectbox("Select their Tenure", (0,1,2,3,4,5,6,7,8,9,10))
    Tenure = st.slider("What is the Clients Tenure", 0, 10,1)

    ##NumOfProducts = st.selectbox("Select the number of products they have purchased", (1,2,3,4))
    NumOfProducts = st.slider("How many Number of Products have they purchased from us", 1, 4,1)

    HasCrCard=st.selectbox("Does the Clent have a Credit Card?",("No","Yes"))


    IsActiveMember =st.selectbox("Is the client an Active member?",("No","Yes"))


    Agechoices = {0:"0-20", 1:"21-25",2:"26-30", 3:"31-35",
    4:"36-40",5:"41-50",6:"51-60",7:"61-70",8:"71-100"}

    # Dropdown input
    Age = st.selectbox("Select the Age Category", Agechoices.keys(), format_func=lambda x:Agechoices[ x ])

    # If button is pressed
    if st.button("Submit"):
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        # Store inputs into dataframe

    
        X = pd.DataFrame([[CreditScore, Geography, Gender, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        Age]], 
                        columns = ['CreditScore', 'Geography', 'Gender', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'age_binned'])
        #st.write("From the following data that you have given: ")
        #st.write(X)
        X = X.replace(["No", "Yes"], [0, 1])

        # Get prediction
        prediction = clf.predict(X)[0]
        predict_probability= clf.predict_proba(X)


        # Output prediction

        if prediction == 1:
            st.write(f'This client is likely to churn. The probability of this client churning is at  {round(predict_probability[0][1]*100 , 2)}%')
        else:
            st.write(f'This client is NOT likely to churn. The probability of this client NOT churning is at  {round(predict_probability[0][0]*100 , 2)}%')
