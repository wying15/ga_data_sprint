import streamlit as st
import pandas as pd
import pickle

# Load the dataset (for column reference)
file_path = 'X_train_transformed.csv'
data = pd.read_csv(file_path)

# Convert object columns to appropriate types (float/int)
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            pass

# Load the PyCaret model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Identifying base columns and dummy variable groups
base_columns = ['floor_area_sqm', 'Tranc_Year', 'hdb_age', 'max_floor_lvl', 
                'total_dwelling_units', 'Mall_Nearest_Distance', 'Mall_Within_500m', 
                'Hawker_Nearest_Distance', 'Hawker_Within_500m', 'mrt_nearest_distance', 
                'bus_stop_nearest_distance', 'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 
                'lower']

# Define a dictionary to store min and max values for each base column
min_max_values = {
    'floor_area_sqm': (0, 300),  
    'Tranc_Year': (2000, 2040),
    'hdb_age': (0, 100),
    'max_floor_lvl': (1, 50),
    'commercial': (0, 1),  
    'market_hawker': (0, 1),  
    'multistorey_carpark': (0, 1),  
    'precinct_pavilion': (0, 1),  
    'total_dwelling_units': (10, 500),
    'Mall_Nearest_Distance': (0, 5000),
    'Mall_Within_500m': (0, 50),  
    'Hawker_Nearest_Distance': (0, 5000),
    'Hawker_Within_500m': (0, 50),  
    'mrt_nearest_distance': (0, 5000),
    'bus_interchange': (0, 1),  
    'mrt_interchange': (0, 1),  
    'bus_stop_nearest_distance': (0, 5000),
    'pri_sch_nearest_distance': (0, 5000),
    'sec_sch_nearest_dist': (0, 5000),
    'affiliation': (0, 1),  
    'flat_type_int': (1, 7),
    'lower': (1, 51),  
}

# Dummy variable groups
town_columns = [col for col in data.columns if col.startswith('town_')]
flat_model_columns = [col for col in data.columns if col.startswith('flat_model_')]

# User input for base columns
st.title("HDB Resale Price Predictor")

# Define a dictionary mapping column names to their default values
default_values = {
    'floor_area_sqm': 100,
    'Tranc_Year': 2024,
    'hdb_age': 25,
    'max_floor_lvl': 50,
    'commercial': 0,
    'market_hawker': 0,
    'multistorey_carpark': 0,
    'precinct_pavilion': 0,
    'total_dwelling_units': 200,
    'Mall_Nearest_Distance': 500,
    'Mall_Within_500m': 1,
    'Hawker_Nearest_Distance': 1000,
    'Hawker_Within_500m': 1,
    'mrt_nearest_distance': 1000,
    'bus_interchange': 0,
    'mrt_interchange': 0,
    'bus_stop_nearest_distance': 500,
    'pri_sch_nearest_distance': 1000,
    'sec_sch_nearest_dist': 1000,
    'affiliation': 1,
    'flat_type_int': 4,
    'lower': 20,
}

# Refined prompts for each base column
refined_prompts = {
    'floor_area_sqm': "Enter the floor area (in square metres) of the unit.",
    'Tranc_Year': "Enter the transaction year (e.g., 2023).",
    'hdb_age': "Enter the age of the HDB flat.",
    'max_floor_lvl': "Enter the maximum floor level of the block.",
    'commercial': "Are there commerical shops within the block?",
    'market_hawker': "Is there a market or hawker center within the same block?",
    'multistorey_carpark': "Is there a multi-storey carpark within the block?",
    'precinct_pavilion': "Is there a precinct pavilion with the block?",
    'total_dwelling_units': "Enter the total number of dwelling units in the block.",
    'Mall_Nearest_Distance': "Enter the distance to the nearest mall (in meters).",
    'Mall_Within_500m': "How many malls are there within 500 meters?",
    'Hawker_Nearest_Distance': "Enter the distance to the nearest hawker center (in meters).",
    'Hawker_Within_500m': "How many hawker centres within 500 meters?",
    'mrt_nearest_distance': "Enter the distance to the nearest MRT station (in meters).",
    'bus_interchange': "Does the nearest MRT station also have a bus interchange?",
    'mrt_interchange': "Is the nearest MRT station also a MRT interchange?",
    'bus_stop_nearest_distance': "Enter the distance to the nearest bus stop (in meters).",
    'pri_sch_nearest_distance': "Enter the distance to the nearest primary school (in meters).",
    'sec_sch_nearest_dist': "Enter the distance to the nearest secondary school (in meters).",
    'affiliation': "Is the flat near a secondary school that has a primary school?",
    'flat_type_int': "Select the flat type (1-7).",
    'lower': "Select the floor level.",
}

inputs = {}
# Get user inputs with refined prompts
for col in base_columns:
    inputs[col] = st.number_input(refined_prompts[col], 
                                 value=default_values.get(col, data[col].median()), 
                                 min_value=min_max_values[col][0], 
                                 max_value=min_max_values[col][1], 
                                 step=1)

# Dropdown options for flat type
flat_type_options = {
    1: "1-Room",
    2: "2-Room",
    3: "3-Room",
    4: "4-Room",
    5: "5-Room",
    6: "Multi-Generation",
    7: "Executive"
}

# Get user input for flat type
inputs['flat_type_int'] = st.selectbox("Select Flat Type", list(flat_type_options.values()))

# Convert selected flat type to integer
inputs['flat_type_int'] = int([k for k, v in flat_type_options.items() if v == inputs['flat_type_int']][0])

# Consolidated user input for 'town' (dummy variable group)
towns = [col.replace('town_', '') for col in town_columns]
selected_town = st.selectbox("Select Town", towns)
inputs['town'] = selected_town

# Consolidated user input for 'flat_model' (dummy variable group)
flat_models = [col.replace('flat_model_', '') for col in flat_model_columns]
default_flat_model_index = 1
selected_flat_model = st.selectbox("Select Flat Model", flat_models)
inputs['flat_model'] = selected_flat_model

# Convert binary columns to yes/no selections
binary_columns = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion', 'bus_interchange', 'mrt_interchange','affiliation']

refined_binary_prompts = {
    'commercial': "Are there commerical shops within the block?",
    'market_hawker': "Is there a market or hawker center within the same block?",
    'multistorey_carpark': "Is there a multi-storey carpark within the block?",
    'precinct_pavilion': "Is there a precinct pavilion within the block?",
    'bus_interchange': "Does the nearest MRT station also have a bus interchange?",
    'mrt_interchange': "Is the nearest MRT station also a MRT interchange?",
    'affiliation': "Is the flat near a secondary school that has a primary school?",
}

for col in binary_columns:
    inputs[col] = st.selectbox(refined_binary_prompts[col], options=['Yes', 'No'])
    inputs[col] = 1 if inputs[col] == 'Yes' else 0

# Prepare the data for the model (dummy encoding)
# We will create a DataFrame with one row, with dummy variables filled in accordingly.
input_data = pd.DataFrame(columns=data.columns)

# Add base column values
for col in base_columns:
    input_data.at[0, col] = inputs[col]

# One-hot encode the selected town and flat model by setting the corresponding dummy variables
for col in town_columns:
    input_data.at[0, col] = 1 if col == f'town_{selected_town}' else 0

for col in flat_model_columns:
    input_data.at[0, col] = 1 if col == f'flat_model_{selected_flat_model}' else 0

# Drop unnecessary columns (like the index column if it exists)
input_data = input_data.drop(columns=['Unnamed: 0'], errors='ignore')

# Convert columns to the correct types (if they aren't already)
input_data = input_data.astype(float)

# Show the prepared input data
#st.write("Prepared Data for Prediction", input_data)

# Button to trigger the prediction
if st.button("Make Prediction"):
    # Make a prediction using the loaded PyCaret model
    prediction = model.predict(input_data)
    
    # Display the prediction result
    st.success(f"The predicted value is: ${'{:0,.2f}'.format(float(prediction[0]))}")

