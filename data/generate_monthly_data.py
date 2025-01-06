import pandas as pd
import os

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data directory
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))

# Define paths to the hourly and monthly data files
markets_hourly_path = os.path.join(data_dir, 'markets.xlsx')
demands_hourly_path = os.path.join(data_dir, 'demands_hourly_year.csv')
demands_monthly_30_path = os.path.join(data_dir, 'demands_monthly_30.xlsx')

try:
    # Use pd.read_excel for Excel files
    markets_hourly = pd.read_excel(markets_hourly_path)
    demands_hourly = pd.read_csv(demands_hourly_path)

except UnicodeDecodeError:
    print("Encoding error. Attempting to load with 'latin1' encoding.")
    markets_hourly = pd.read_excel(markets_hourly_path, encoding='latin1')
    demands_hourly = pd.read_csv(demands_hourly_path, encoding='latin1')

# (Continue with the rest of the script, including datetime indexing and data resampling)

# Function to ensure datetime index
def ensure_datetime_index(df, df_name, datetime_column_name='time'):
    if datetime_column_name in df.columns:
        df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])
        df.set_index(datetime_column_name, inplace=True)
        print(f"{df_name} datetime index set from existing '{datetime_column_name}' column.")
    else:
        # If datetime column doesn't exist, create one assuming hourly data starting from a specific date
        start_date = pd.to_datetime('2021-01-01 00:00')
        df.index = pd.date_range(start=start_date, periods=len(df), freq='H')
        print(f"Note: '{df_name}' did not have a datetime column. A datetime index has been created starting from {start_date}.")

# Apply the function to both dataframes
ensure_datetime_index(markets_hourly, 'markets_hourly')
ensure_datetime_index(demands_hourly, 'demands_hourly')

# Resample to monthly data
# For demands (e.g., energy consumption), sum over the month
demands_monthly = demands_hourly.resample('M').sum()

# Repeat the 12-month data to create 30 years of monthly data
demands_monthly_30 = pd.concat([demands_monthly] * 30, ignore_index=True)

# Create a new datetime index for 30 years, starting from the first month in 'demands_monthly'
start_date = demands_monthly.index[0]
demands_monthly_30.index = pd.date_range(start=start_date, periods=len(demands_monthly_30), freq='M')

# Reset the index and rename to 'Month' for clarity
demands_monthly_30.reset_index(inplace=True)
demands_monthly_30.rename(columns={'index': 'Month'}, inplace=True)

# Save the 30-year data to Excel
with pd.ExcelWriter(demands_monthly_30_path, date_format='YYYY-MM') as writer:
    demands_monthly_30.to_excel(writer, index=False)

print("30-year monthly data has been saved to:")
print(f"Demands 30-year monthly data: {demands_monthly_30_path}")
