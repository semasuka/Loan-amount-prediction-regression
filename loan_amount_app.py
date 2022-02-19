import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st
import boto3
import tempfile
import json
import requests
from streamlit_lottie import st_lottie_spinner


full_data = pd.read_csv(
    "https://raw.githubusercontent.com/semasuka/Loan-amount-prediction-regression/main/datasets/train.csv"
)


# split the data into train and test
def data_split(df, test_size):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()


####################### Classes used to preprocess the data ##############################


class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=['Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Property Age', 'Property Price']):
        self.feat_with_outliers = feat_with_outliers

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(.25)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            # keep the data within 3 IQR
            df = df[~((df[self.feat_with_outliers] < (Q1 - 1.5 * IQR)) |
                      (df[self.feat_with_outliers] > (Q3 + 1.5 * IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_imputed_ft=['Gender', 'Income Stability', 'Dependents', 'Has Active Credit Card', 'Property Location'], median_imputed_ft=['Income (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Property Age']):
        self.mode_imputed_ft = mode_imputed_ft
        self.median_imputed_ft = median_imputed_ft

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.mode_imputed_ft + self.median_imputed_ft).issubset(df.columns)):
            # drop missing values in the target feature
            df.dropna(inplace=True, axis=0, subset=[
                      'Loan Sanction Amount (USD)'])
            # impute missing values with mode
            for ft in self.mode_imputed_ft:
                the_mode = df[ft].mode()[0]
                df[ft] = df[ft].fillna(the_mode)
            # impute missing values with median
            for ft in self.median_imputed_ft:
                the_median = df[ft].median()
                df[ft] = df[ft].fillna(the_median)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class DropUncommonProfession(BaseEstimator, TransformerMixin):
    def __init__(self, profession_list=['Student', 'Unemployed', 'Businessman']):
        self.profession_list = profession_list

    def fit(self, df):
        return self

    def transform(self, df):
        if ('Profession' in df.columns):
            # only keep the professions that are not in the profession_list
            df = df[~df['Profession'].isin(self.profession_list)]
            return df
        else:
            print("Profession feature is not in the dataframe")
            return df


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['Customer ID', 'Name', 'Type of Employment', 'Property ID']):
        self.feature_to_drop = feature_to_drop

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class ValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_999_val=['Co-Applicant', 'Current Loan Expenses (USD)', 'Loan Sanction Amount (USD)', 'Property Price']):
        self.feat_with_999_val = feat_with_999_val

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.feat_with_999_val).issubset(df.columns)):
            for ft in self.feat_with_999_val:
                # replace any occurance of -999.000 with 0
                df[ft].replace(-999.000, 0, inplace=True, regex=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=['Age', 'Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Property Age', 'Property Price']):
        self.min_max_scaler_ft = min_max_scaler_ft

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(
                df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=['Gender', 'Profession', 'Location', 'Expense Type 1', 'Expense Type 2', 'Has Active Credit Card', 'Property Location', 'Income Stability']):
        self.one_hot_enc_ft = one_hot_enc_ft

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # function to one hot encode the features in one_hot_enc_ft
            def one_hot_enc(df, one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                # get the result of the one hot encoding columns names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(
                    one_hot_enc_ft)
                # change the array of the one hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(
                ), columns=feat_names_one_hot_enc, index=df.index)
                return df
            # function to concatenat the one hot encoded features with the rest of features that were not encoded

            def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
                # get the rest of the features
                rest_of_features = [
                    ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat(
                    [one_hot_enc_df, df[rest_of_features]], axis=1)
                return df_concat
            # one hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
            # returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(
                df, one_hot_enc_df, self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, col_with_skewness=['Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Property Age']):
        self.col_with_skewness = col_with_skewness

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.col_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.col_with_skewness] = np.cbrt(df[self.col_with_skewness])
            return df
        else:
            print('One or more skewed columns are not found')
            return df


def full_pipeline(df):
    pipeline = Pipeline([
        ('drop features', DropFeatures()),
        ('outlier remover', OutlierImputer()),
        ('drop uncommon profession', DropUncommonProfession()),
        ('missing value imputer', MissingValueImputer()),
        ('-999 value imputer', ValueImputer()),
        ('skewness handler', SkewnessHandler()),
        ('min max scaler', MinMaxWithFeatNames()),
        ('one hot encoder', OneHotWithFeatNames())
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep


############################# Streamlit interface ############################


st.write("""
# Approved loan amount prediction
This app predicts how much will be granted to a loan applicant. Just fill in the following information and click on the Predict button.
""")


# Gender input
st.write("""
## Gender
""")
input_gender = st.radio('Select you gender', ['Male', 'Female'], index=0)


# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select your age', value=40,
                      min_value=18, max_value=65, step=1)


# Income
st.write("""
## Income
""")
input_income = st.slider('Select your income (biweekly)', value=2500,
                         min_value=0, max_value=5500, step=10)


# Income stability
st.write("""
## Income stability
""")
income_stab = st.radio('How is your income stability',
                       ['Low', 'High'], index=0)


# Profession dropdown
st.write("""
## Profession
""")
professions = ['Working', 'Commercial associate', 'Pensioner', 'State servant']
input_professions = st.selectbox(
    'Select your profession', professions)


# Residence location dropdown
st.write("""
## Residence location
""")
locations = ['Semi-Urban', 'Rural', 'Urban']
location = st.selectbox(
    'Select your residence location', locations)


# Current loan expenses
st.write("""
## Current loan expenses
""")
input_current_loan_amt = st.slider('Select your current loan expenses', value=0,
                                   min_value=0, max_value=1000, step=10)


# Expenses type 1
st.write("""
## Expenses type 1
""")
exp_type_one = st.radio('Do you have expenses type one?', [
                        'Yes', 'No'], index=1)


# Expenses type 2
st.write("""
## Expenses type 2
""")
exp_type_two = st.radio('Do you have expenses type two?', [
                        'Yes', 'No'], index=1)


# Number of dependents
st.write("""
## Number of dependents
""")
dependents_count = st.slider('How many dependents do you have?', value=0,
                             min_value=0, max_value=4, step=1)


# Credit score
st.write("""
## Credit score
""")
credit_score = st.slider('Select your credit score', value=740,
                         min_value=580, max_value=900, step=1)


# Loan default
st.write("""
## Loan default
""")
loan_default_dict = {'Yes': 1, 'No': 0}
loan_default_input = st.radio('Have you ever had a loan default', [
    'Yes', 'No'], index=1)
loan_default_input_val = loan_default_dict.get(loan_default_input)

# Has a credit card
st.write("""
## Credit card
""")
cc_status = ['Active', 'Inactive', 'Unpossessed']
cc_status_input = st.selectbox(
    'What is the status of your credit card', cc_status)


# Property age
st.write("""
## Property age
""")
property_age = st.slider('Select the property age', value=5,
                         min_value=1, max_value=14, step=1) * 365.25


# Property price
st.write("""
## Property price
""")
prop_price = st.slider('Select your property price', value=0,
                       min_value=0, max_value=355000, step=100)


# Property type
st.write("""
## Property type
""")
property_type = [1, 2, 3, 4]
property_type_input = st.selectbox(
    'Select the property type', property_type)


# Property location dropdown
st.write("""
## Property location
""")
prop_locations = ['Semi-Urban', 'Rural', 'Urban']
prop_location = st.selectbox(
    'Select your property location', prop_locations)


# Co-applicant
st.write("""
## Co-applicant
""")
co_applicant_dict = {'Yes': 1, 'No': 0}
co_applicant = st.radio('Do you have a co-applicant?', [
                        'Yes', 'No'], index=1)
co_applicant_val = co_applicant_dict.get(co_applicant)


# Loan amount requested
st.write("""
## Loan amount requested
""")
loan_amount_req = st.slider('Select your current loan expenses', value=0,
                            min_value=0, max_value=240000, step=100)

st.markdown('##')
st.markdown('##')
# Button
predict_bt = st.button('Predict')
st.markdown('##')
st.markdown('##')

# list of all the inputs
profile_to_predict = [
    '',  # customer id
    '',  # name
    input_gender[:1],
    input_age,
    input_income,
    income_stab,
    input_professions,
    '',  # type of employment
    location,
    loan_amount_req,
    input_current_loan_amt,
    exp_type_one[:1],
    exp_type_two[:1],
    dependents_count,
    credit_score,
    loan_default_input_val,
    cc_status_input,
    0,  # property id
    property_age,
    property_type_input,
    prop_location,
    co_applicant_val,
    prop_price,
    0  # loan amount sanctioned
]

# Convert to dataframe with column names
profile_to_predict_df = pd.DataFrame(
    [profile_to_predict], columns=train_copy.columns)


# add the profile to predict as a last row in the train data
train_copy_with_profile_to_pred = pd.concat(
    [train_copy, profile_to_predict_df], ignore_index=True)


# whole dataset prepared
profile_to_pred_prep = full_pipeline(
    train_copy_with_profile_to_pred).tail(1).drop(columns=['Loan Sanction Amount (USD)'])


# Animation function
@st.experimental_memo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_loading_an = load_lottieurl(
    'https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json')


def make_prediction():
    # connect to s3 bucket
    # for s3 API keys when deployed on streamlit share
    client = boto3.client(
        's3', aws_access_key_id=st.secrets["access_key"], aws_secret_access_key=st.secrets["secret_access_key"])
    # for s3 API keys when deployed on locally
    # client = boto3.client('s3', aws_access_key_id='access_key',
    #                       aws_secret_access_key='secret_access_key')

    bucket_name = "loanamount"
    key = "trained_Random Forest Regression"

    # load the model from s3 in a temporary file
    with tempfile.TemporaryFile() as fp:
        client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
        fp.seek(0)
        model = joblib.load(fp)

    # prediction from the model on AWS S3
    return model.predict(profile_to_pred_prep)


if predict_bt:
    with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
        final_pred = make_prediction()
    # if final_pred exists, then stop displaying the loading animation
    st.info(
        '## You have been approved for a loan amount of {0:.2f} USD'.format(final_pred[0]))
    st.balloons()
