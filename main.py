import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Define Chinese New Year months for each year
chinese_new_year_months = {
    2019: [2],
    2020: [1],
    2021: [2],
    2022: [2],
    2023: [1],
    2024: [2],
    2025: [2],
    2026: [2],
    2027: [2],
    2028: [1],
    2029: [2],
    2030: [2]
    
}

# Data preprocessing and feature engineering function
def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    data['year'] = data['date'].dt.year
    data['month_order'] = (data['year'] - data['year'].min()) * 12 + data['month']
    data['chinese_new_year_actual'] = 0
    data['chinese_new_year_prev_1'] = 0
    data['chinese_new_year_prev_2'] = 0
    data['chinese_new_year_following'] = 0
    
    for year, months in chinese_new_year_months.items():
        for month in months:
            year_month = str(year) + '-' + str(month).zfill(2)
            data.loc[data['date'].dt.strftime('%Y-%m') == year_month, 'chinese_new_year_actual'] = 1

            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            prev_year_month = str(prev_year) + '-' + str(prev_month).zfill(2)
            data.loc[data['date'].dt.strftime('%Y-%m') == prev_year_month, 'chinese_new_year_prev_1'] = 1

            prev_prev_month = month - 2 if month > 2 else 12 + month - 2
            prev_prev_year = year if month > 2 else year - 1
            prev_prev_year_month = str(prev_prev_year) + '-' + str(prev_prev_month).zfill(2)
            data.loc[data['date'].dt.strftime('%Y-%m') == prev_prev_year_month, 'chinese_new_year_prev_2'] = 1

            following_month = month + 1 if month < 12 else 1
            following_year = year if month < 12 else year + 1
            following_year_month = str(following_year) + '-' + str(following_month).zfill(2)
            data.loc[data['date'].dt.strftime('%Y-%m') == following_year_month, 'chinese_new_year_following'] = 1

    return data

# Function to convert wide format data to long format
def convert_to_long_format(data):
    required_columns = ['Country','Region', 'Material',  'Material Description']
    if not all(col in data.columns for col in required_columns):
        st.error("Missing required columns in the data.")
        return None
    date_columns = [col for col in data.columns if col not in required_columns]
    # Concatenate Material Description and Material columns
    data['Product'] = data['Material Description'] + '-' + data['Material'].astype(str)
    return pd.melt(data, id_vars=['Country', 'Region', 'Product', 'Material',  'Material Description'], value_vars=date_columns, var_name='date', value_name='sales')

# Function to train and evaluate models
def train_evaluate_model(train_data, test_data, future_model_type):
    if future_model_type == "XGBoost":
        model = XGBRegressor(random_state=42, n_estimators=200)
    elif future_model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42, n_estimators=200)
    elif future_model_type == "ETS":
        model = ExponentialSmoothing(
            train_data['sales'],
            seasonal='add',
            seasonal_periods=12  # Adjust as per your data seasonality
        )

    if future_model_type != "ETS":
        X_train = train_data[['month', 'quarter', 'year', 'month_order', 'chinese_new_year_actual', 'chinese_new_year_prev_1', 'chinese_new_year_prev_2', 'chinese_new_year_following']]
        y_train = train_data['sales']
        X_test = test_data[['month', 'quarter', 'year', 'month_order', 'chinese_new_year_actual', 'chinese_new_year_prev_1', 'chinese_new_year_prev_2', 'chinese_new_year_following']]
        y_test = test_data['sales']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = pd.DataFrame({
            'Date': test_data['date'],
            'Actuals': y_test.astype(float),
            'Forecast': y_pred.astype(float),
            'Dataset': ['Test'] * len(test_data)  # Add a column to label the test data
        })
        
        train_pred = model.predict(X_train)
        train_result = pd.DataFrame({
            'Date': train_data['date'],
            'Actuals': y_train.astype(float),
            'Forecast': train_pred.astype(float),
            'Dataset': ['Train'] * len(train_data)  # Add a column to label the train data
        })

    else:  # ETS model case
        model = model.fit()
        y_pred = model.forecast(len(test_data))  # Adjust for the forecast length as per your needs

        result = pd.DataFrame({
            'Date': test_data['date'],
            'Actuals': test_data['sales'].values,
            'Forecast': y_pred.values,
            'Dataset': ['Test'] * len(test_data)  # Add a column to label the test data
        })
        
        train_result = pd.DataFrame({
            'Date': train_data['date'],
            'Actuals': train_data['sales'].values,
            'Forecast': model.fittedvalues.values,
            'Dataset': ['Train'] * len(train_data)  # Add a column to label the train data
        })
            
    result_combined = pd.concat([train_result, result]).reset_index(drop=True)

    result['Abs_error'] = abs(result['Actuals'] - result['Forecast'])
    result['APE'] = result['Abs_error'] / result['Actuals'] * 100

    # Calculate WAPE and Hit Rate for all data
    sum_abs_error = np.sum(result['Abs_error'])
    sum_actuals = np.sum(result['Actuals'])
    wape = sum_abs_error / sum_actuals

    count_greater_than_threshold = result[result['APE'] < threshold].shape[0]
    total_values = result['APE'].shape[0]
    hit_rate = (count_greater_than_threshold / total_values) * 100
    
    combined_score = 0.5 * (1 - wape) + 0.5 * hit_rate

    return model, result, result_combined, wape, hit_rate, combined_score

# Function to plot decomposition
def plot_decomposition(data, title='Decomposition of Sales Data'):
    result_dec = seasonal_decompose(data['sales'], model='additive', period=12)  # Adjust period according to your data frequency
    
    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    result_dec.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result_dec.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result_dec.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    result_dec.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.xlabel('Date')
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


# Initialize Streamlit
st.set_page_config(layout="wide")
st.title('Advanced Forecasting Tool')

# Sidebar for data upload and parameter selection
uploaded_file = st.sidebar.file_uploader("Upload Historical Data", type=["csv", "xlsx"])
data_long = None

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Convert data to long format and display columns
    data_long = convert_to_long_format(data)
    if data_long is not None:
        data_long = preprocess_data(data_long)
        
        # Let user select Country, Region and Material Description
        country_options = data_long['Country'].unique()
        region_options = data_long['Region'].unique()
        selected_country = st.sidebar.selectbox('Select Country', country_options)
        selected_region = st.sidebar.selectbox('Select Region', region_options)
        material_descriptions = data_long[(data_long['Country'] == selected_country) & (data_long['Region'] == selected_region)]['Product'].unique()
        #selected_material_descriptions = st.sidebar.multiselect('Select Material Descriptions', material_descriptions, material_descriptions[:8])
        # Add a CSS snippet to make the multiselect widget scrollable
        st.markdown("""
            <style>
                .stMultiSelect [role="listbox"] {
                    max-height: 200px;
                }
            </style>
            """, unsafe_allow_html=True)

        selected_material_descriptions = st.sidebar.multiselect('Select Material Descriptions', material_descriptions, material_descriptions[:500])

        # Let user select date range and model for future prediction
        st.sidebar.subheader("Future Prediction Settings")
        slider_key = "future_date_range_slider"
        future_date_range = st.sidebar.slider("Select the number of months for future prediction", 12, 18, 12, key=slider_key)
        future_model_type = st.sidebar.selectbox("Select model for future prediction", ["XGBoost", "Random Forest", "ETS", "Best Model"])

        all_predictions = []
        all_history = []

        # Summary tables
        wape_summary = []
        hit_rate_summary = []
        best_model_summary = []

      
              
                
        for selected_material_description in selected_material_descriptions:
            with st.expander(f"Analysis for {selected_material_description} ({selected_country} - {selected_region})", expanded=False):
                #st.subheader(f"Sales Trend for {selected_material_description}") 
                

                # Filter data based on selected Country, Region, and Material Description
                filtered_data = data_long[(data_long['Product'] == selected_material_description) & 
                                          (data_long['Country'] == selected_country) & 
                                          (data_long['Region'] == selected_region)]
                selected_material_id = filtered_data['Material'].unique()[0]

                # Split data in train (80%) and test (20%)
                split_index = int(len(filtered_data) * 0.8)

                # Sort data by date to maintain temporal order
                filtered_data = filtered_data.sort_values(by='date')

                # Split data into training and testing sets
                train_data = filtered_data.iloc[:split_index]
                test_data = filtered_data.iloc[split_index:]

                fig = px.line(filtered_data, x='date', y='sales', title=f'Sales Trend for {selected_material_description}')
                st.plotly_chart(fig)
                
                
                fig = plot_decomposition(filtered_data)
                st.pyplot(fig)

                # Train models and display results for all models
                models = []
                wape_scores = []
                hit_rates = []
                combined_scores = []
                threshold = 30  # Define threshold here

                for model_type in ["XGBoost", "Random Forest", "ETS"]:
                    model, result, result_combined, wape, hit_rate, combined_score  = train_evaluate_model(train_data, test_data, model_type)

                    st.subheader(f"Results for {model_type}")
                    st.write(f"WAPE: {wape*100:.0f}%")
                    st.write(f"Hit Rate (APE < {threshold}%): {hit_rate:.0f}%")

                    
                    # Plotting actuals vs forecast
                    
                    # Separate data
                    train_actuals = result_combined[result_combined['Dataset'] == 'Train']
                    train_forecast = train_actuals.copy()
                    train_forecast['Actuals'] = None

                    test_actuals = result_combined[result_combined['Dataset'] == 'Test']
                    test_forecast = test_actuals.copy()
                    test_forecast['Actuals'] = None

                    # Plotting actuals vs forecast with 4 different colors
                    fig = px.line(title=f"Actual vs Forecasted Sales for {selected_material_description} ({selected_country} - {selected_region})")

                    # Add traces for each category
                    fig.add_scatter(x=train_actuals['Date'], y=train_actuals['Actuals'], mode='lines', name='Train Actuals', line=dict(color='blue'))
                    fig.add_scatter(x=train_forecast['Date'], y=train_forecast['Forecast'], mode='lines', name='Train Forecast', line=dict(color='green'))
                    fig.add_scatter(x=test_actuals['Date'], y=test_actuals['Actuals'], mode='lines', name='Test Actuals', line=dict(color='orange'))
                    fig.add_scatter(x=test_forecast['Date'], y=test_forecast['Forecast'], mode='lines', name='Test Forecast', line=dict(color='red'))

                    # Show the plot in Streamlit
                    st.plotly_chart(fig)
                    
                    models.append(model)
                    wape_scores.append(wape)
                    hit_rates.append(hit_rate)
                    combined_scores.append(combined_score)

                    # Append results to summary tables
                    wape_summary.append({
                        'Country': selected_country,
                        'Region': selected_region,
                        'Material': str(selected_material_id),
                        'Material Description': selected_material_description,
                        f'{model_type} WAPE': (wape * 100).round()
                    })

                    hit_rate_summary.append({
                        'Country': selected_country,
                        'Region': selected_region,
                        'Material': str(selected_material_id),
                        'Material Description': selected_material_description,
                        f'{model_type} HitRate': round(hit_rate,0)
                    })

                # Determine best model for each material
                best_model_idx = np.argmax(combined_scores)
                best_model_type = ["XGBoost", "Random Forest", "ETS"][best_model_idx]
                best_model = models[best_model_idx]

                # Add best model to the summary table
                best_model_summary.append({
                    'Country': selected_country,
                    'Region': selected_region,
                    'Material': str(selected_material_id),
                    'Material Description': selected_material_description,
                    'Best Model': best_model_type
                })

                # Determine the model for future predictions
                if future_model_type == "Best Model":
                    future_model = best_model
                    future_model_name = best_model_type
                else:
                    future_model = models[["XGBoost", "Random Forest", "ETS"].index(future_model_type)]
                    future_model_name = future_model_type

                future_dates = pd.date_range(start=filtered_data['date'].max(), periods=future_date_range + 1, freq='M')[1:]
                future_data = pd.DataFrame({'date': future_dates})
                future_data['month'] = future_data['date'].dt.month
                future_data['quarter'] = future_data['date'].dt.quarter
                future_data['year'] = future_data['date'].dt.year

                # Generate month_order for historical and future data
                last_month_order = filtered_data['month_order'].max()
                future_data['month_order'] = last_month_order + np.arange(1, future_date_range + 1)

                for year, months in chinese_new_year_months.items():
                    for month in months:
                        year_month = str(year) + '-' + str(month).zfill(2)
                        future_data.loc[future_data['date'].dt.strftime('%Y-%m') == year_month, 'chinese_new_year_actual'] = 1

                        prev_month = month - 1 if month > 1 else 12
                        prev_year = year if month > 1 else year - 1
                        prev_year_month = str(prev_year) + '-' + str(prev_month).zfill(2)
                        future_data.loc[future_data['date'].dt.strftime('%Y-%m') == prev_year_month, 'chinese_new_year_prev_1'] = 1

                        prev_prev_month = month - 2 if month > 2 else 12 + month - 2
                        prev_prev_year = year if month > 2 else year - 1
                        prev_prev_year_month = str(prev_prev_year) + '-' + str(prev_prev_month).zfill(2)
                        future_data.loc[future_data['date'].dt.strftime('%Y-%m') == prev_prev_year_month, 'chinese_new_year_prev_2'] = 1

                        following_month = month + 1 if month < 12 else 1
                        following_year = year if month < 12 else year + 1
                        following_year_month = str(following_year) + '-' + str(following_month).zfill(2)
                        future_data.loc[future_data['date'].dt.strftime('%Y-%m') == following_year_month, 'chinese_new_year_following'] = 1

                future_data.fillna(0, inplace=True)

                # Fit the best model on the entire dataset (train + test) and generate future predictions
                combined_data = pd.concat([train_data, test_data])

                if future_model_name != "ETS":
                    X_future = future_data[['month', 'quarter', 'year', 'month_order', 'chinese_new_year_actual', 'chinese_new_year_prev_1', 'chinese_new_year_prev_2', 'chinese_new_year_following']]
                    X_combined = combined_data[['month', 'quarter', 'year', 'month_order', 'chinese_new_year_actual', 'chinese_new_year_prev_1', 'chinese_new_year_prev_2', 'chinese_new_year_following']]
                    y_combined = combined_data['sales']
                    future_model = future_model.fit(X_combined, y_combined)
                    future_predictions = future_model.predict(X_future)
                else:
                    future_model = ExponentialSmoothing(
                                combined_data['sales'],
                                seasonal='add',
                                seasonal_periods=12  # Adjust as per your data seasonality
                                )
                    future_model = future_model.fit()
                    future_predictions = future_model.forecast(len(future_data))

                # Create a result dataframe with Country, Material Description, and future predictions
                future_result = pd.DataFrame({
                    'Country': [selected_country] * len(future_data),
                    'Region': [selected_region] * len(future_data),
                    'Material': [str(selected_material_id)] * len(future_data),
                    'Material Description': [selected_material_description] * len(future_data),
                    'Date': future_data['date'].values,
                    'Prediction': np.maximum(future_predictions, 0).round()
                })
                future_result['Date'] = pd.to_datetime(future_result['Date']).dt.strftime('%Y-%m')

                all_predictions.append(future_result)
             
                # Display future predictions
                st.subheader(f"Future Predictions for {selected_material_description}  ({selected_country} - {selected_region})")
                st.write(future_result)
                
                # Plot historical data and future predictions
                # Combine historical and future data
                historical_data = filtered_data.copy()
                historical_data['Type'] = 'Historical'
                future_result['Type'] = 'Future'
                combined_data = pd.concat([historical_data[['date', 'sales', 'Type']], future_result[['Date', 'Prediction', 'Type']].rename(columns={'Date': 'date', 'Prediction': 'sales'})])

                # Create the plot
                fig = px.line(combined_data, x='date', y='sales', color='Type', labels={'date': 'Date', 'sales': 'Sales'})
                fig.update_layout(
                    #title=f"Sales Forecast for {selected_material_description} ({selected_country} - {selected_region})",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    legend_title="Data Type"
                )

                # Display the plot
                st.plotly_chart(fig)
                
                # Create a historical data DataFrame
                historical_data = filtered_data[['Country', 'Region', 'Material','Material Description', 'date', 'sales']]
                # Rename the columns
                historical_data = historical_data.rename(columns={
                    'date': 'Date',
                    'sales': 'Sales'
                })
                historical_data['Date'] = historical_data['Date'].dt.strftime('%Y-%m')
                historical_data['Material'] = historical_data['Material'].astype(str)
                historical_data['Sales'] = historical_data['Sales'].round(0)
                all_history.append(historical_data)
        
        
        # Combine all predictions into a single dataframe and display as a table
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        # Pivot combined_predictions to have dates as columns
        pivot_predictions = combined_predictions.pivot(index=['Country','Region','Material', 'Material Description'], columns='Date', values='Prediction')

        # Reset index to flatten the multi-index columns
        pivot_predictions = pivot_predictions.reset_index()
        
        
        # Combine all predictions into a single dataframe and display as a table
        combined_history = pd.concat(all_history, ignore_index=True)
        # Pivot combined_predictions to have dates as columns
        pivot_history = combined_history.pivot(index=['Country','Region','Material', 'Material Description'], columns='Date', values='Sales')

        # Reset index to flatten the multi-index columns
        pivot_history = pivot_history.reset_index()
        
        
        

        # Display summary tables for WAPE, Hit Rate, and Best Model

        st.subheader("Best Model Summary Table")
        best_model_summary_df = pd.DataFrame(best_model_summary)
        st.write(best_model_summary_df)
        
        st.subheader("WAPE Summary Table")
        wape_summary_df = pd.DataFrame(wape_summary).groupby(['Country','Region', 'Material', 'Material Description']).mean().reset_index()
        st.write(wape_summary_df)

        st.subheader("Hit Rate Summary Table")
        hit_rate_summary_df = pd.DataFrame(hit_rate_summary).groupby(['Country','Region','Material', 'Material Description']).mean().reset_index()
        st.write(hit_rate_summary_df)
        
        st.subheader("Combined Future Predictions")
        st.write(pivot_predictions)
        #st.write(pivot_history)
        
        # Merge pivot_predictions and pivot_history on the common columns
        pivot_predictions = pivot_predictions.drop('Material Description', axis=1)
        merged_data = pd.merge(pivot_history, pivot_predictions, on=['Country', 'Region', 'Material'], how='inner', suffixes=('_Historical', '_Predicted'))

        # Display the merged table
        st.subheader("Combined Historical and Future Predictions")
        st.write(merged_data)