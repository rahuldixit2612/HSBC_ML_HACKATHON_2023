import pandas as pd
import pickle as pkl

def process_train_data(train_data, train_helper_data):
    # Remove specified columns from train_data
    train_data.drop(["V1", "V2", "V7", "V8", "V9", "V10", "V11", "V12"], axis=1, inplace=True)
    
    # Remove specified columns from train_helper_data
    train_helper_data.drop(["V1", "V14", "V15"], axis=1, inplace=True)
    
    # Merge train_data and train_helper_data on V3 column (left join)
    merged_data = pd.merge(train_data, train_helper_data, on="V3", how="left")
    
    # Remove columns V3 from merged_data
    # merged_data.drop("V3", axis=1, inplace=True)
    
    # Load ASN IP mapping and country IP mapping data
    asn_ip_mapping = pkl.load(open("asn_mapping.pkl", "rb"))
    country_ip_mapping = pkl.load(open("country_mapping.pkl", "rb"))
    
    # Map country and ASN using IP mapping
    merged_data["country_mapping"] = merged_data["V17"].map(country_ip_mapping)
    merged_data["asn_mapping"] = merged_data["V17"].map(asn_ip_mapping)
    
    # Remove columns V20 and V21 from merged_data
    merged_data.drop(["V20", "V21"], axis=1, inplace=True)
    
    # Convert timestamp columns to datetime format
    merged_data['V5'] = pd.to_datetime(merged_data['V5'])
    merged_data['V13'] = pd.to_datetime(merged_data['V13'])
    merged_data['V16'] = pd.to_datetime(merged_data['V16'])
    
    # Extract hour and minute for each timestamp column
    merged_data['V5_hour'] = merged_data['V5'].dt.hour
    merged_data['V5_minute'] = merged_data['V5'].dt.minute
    merged_data['V13_hour'] = merged_data['V13'].dt.hour
    merged_data['V13_minute'] = merged_data['V13'].dt.minute
    merged_data['V16_hour'] = merged_data['V16'].dt.hour
    merged_data['V16_minute'] = merged_data['V16'].dt.minute
    
    # Extract other time-related features
    merged_data['V5_day_of_week'] = merged_data['V5'].dt.dayofweek
    merged_data['V5_day_of_month'] = merged_data['V5'].dt.day
    merged_data['V5_month'] = merged_data['V5'].dt.month
    merged_data['V5_year'] = merged_data['V5'].dt.year
    
    merged_data['V13_day_of_week'] = merged_data['V13'].dt.dayofweek
    merged_data['V13_day_of_month'] = merged_data['V13'].dt.day
    merged_data['V13_month'] = merged_data['V13'].dt.month
    merged_data['V13_year'] = merged_data['V13'].dt.year
    
    merged_data['V16_day_of_week'] = merged_data['V16'].dt.dayofweek
    merged_data['V16_day_of_month'] = merged_data['V16'].dt.day
    merged_data['V16_month'] = merged_data['V16'].dt.month
    merged_data['V16_year'] = merged_data['V16'].dt.year
    
    # Drop the original timestamp columns if desired
    merged_data.drop(['V5', 'V13', 'V16'], axis=1, inplace=True)

    import pickle

    # Load the saved imputer
    with open('imputer_07_mode.pkl', 'rb') as file:
        imputer = pickle.load(file)

    # Columns to impute in merged_test_data
    columns_to_impute_test = ['V13_day_of_week', 'V13_day_of_month', 'V13_month', 'V13_year',
                            'V16_day_of_week', 'V16_day_of_month', 'V16_month', 'V16_year',
                            'country_mapping', 'asn_mapping']

    # Fill missing values in merged_test_data using the loaded imputer
    merged_data[columns_to_impute_test] = imputer.transform(merged_data[columns_to_impute_test])





    with open('imputer_07_mean.pkl', 'rb') as file:
        imputer = pickle.load(file)

    # Columns to impute in merged_test_data
    columns_to_impute_testt = ['V18', 'V16_hour', 'V16_minute', 'V19', 'V13_hour', 'V13_minute']

    # Fill missing values in merged_test_data using the loaded imputer
    merged_data[columns_to_impute_testt] = imputer.transform(merged_data[columns_to_impute_testt])


    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(merged_data['country_mapping'])
    encoded_dataa = encoded_data[:len(merged_data['country_mapping'])]
    merged_data['country_mapping_encoded'] = encoded_dataa
    merged_data.drop('country_mapping', axis=1, inplace=True)

    cols_to_remove_train = ["V17"]
    merged_data.drop(cols_to_remove_train, axis=1, inplace=True)
 
        
    # Return the processed merged_data
    return merged_data