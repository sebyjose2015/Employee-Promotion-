 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('/content/Employee Promotion Data')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('iStock-career-growth-concept-930x558.jpg')
    image_office = Image.open('promotional-analysis.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to Predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting employee promotion")
    if add_selectbox == 'Online':
        employee_id=st.number_input('employee_id' , min_value=1, max_value=	78298, value=1)
        age =st.number_input('age',min_value=20, max_value=60, value=20)
        length_of_service = st.number_input('length_of_service', min_value=1, max_value=37, value=1)
        previous_year_rating = st.number_input('previous_year_rating', min_value=1.0, max_value=5.0, value=1.0)
        avg_training_score=st.number_input('avg_training_score', min_value=39, max_value=99, value=39)
        gender = st.selectbox('gender', ['m', 'f'])
        is_promoted = st.selectbox('is_promoted', ['0', '1'])
        awards_won? = st.selectbox('awards_won?', ['0', '1'])
        recruitment_channel = st.selectbox('recruitment_channel', ['sourcing', 'referred','other'])
        region = st.selectbox('region', ['region_2', 'region_22','region_7','region_15','region_13','region_26','region_31','region_4','region_27','region_16','Other values'])
        department = st.selectbox('department', ['Sales & Marketing','Operations','Technology','Procurement','Analytics','Other values'])
        output=""
        input_dict={'employee_id':employee_id,'age':age,'length_of_service':length_of_service,'previous_year_rating':previous_year_rating,'avg_training_score':avg_training_score,'is_promoted':is_promoted,'gender' : gender,'awards_won?':awards_won?,'recruitment_channel':recruitment_channel,'region':region,'department':department}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
