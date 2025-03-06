import pandas as pd
import numpy as np
import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def initialize_dataframe(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')  # Hourly frequency
    df = pd.DataFrame(index=date_range)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Hour'] = df.index.hour
    return df

# NEW HELPER FUNCTION: calculates effective growth years, capped by a stabilization year if provided.
def effective_growth_years(df, config, commodity):
    # For hydrogen, use the existing "hydrogen_decline_cutoff_year"
    if commodity == 'hydrogen':
        cutoff = config.get('hydrogen_decline_cutoff_year')
    else:
        # For other commodities, look for a stabilization key, e.g., "electricity_stabilization_year"
        cutoff = config.get(f"{commodity}_stabilization_year")
    if cutoff is None:
         return df['Years Since Start']
    else:
         # Calculate capped years: if the stabilization year is reached, further growth stops.
         return np.minimum(df['Years Since Start'], cutoff - config['start_year'])

def apply_annual_growth(df, config):
    df['Years Since Start'] = df['Year'] - config['start_year']
    
    # Electricity Annual Change
    df['Electricity Annual Change'] = config['electricity_annual_growth']
    ev_start_year = config['electricity_ev_start_year']
    df.loc[df['Year'] >= ev_start_year, 'Electricity Annual Change'] += config['electricity_ev_increase']
    
    # Renewable Adjustment for Electricity
    renewable_start_year = config['renewable_capacity_increase']['start_year']
    df.loc[df['Year'] >= renewable_start_year, 'Electricity Annual Change'] += config['renewable_capacity_increase']['price_adjustment']
    
    # Biomass Annual Change
    df['Biomass Annual Change'] = config['biomass_annual_change']
    biomass_demand_increase_start_year = config['biomass_demand_increase_start_year']
    df.loc[df['Year'] >= biomass_demand_increase_start_year, 'Biomass Annual Change'] += config['biomass_additional_annual_increase']
    
    # Apply Annual Growth with stabilization:
    df['Electricity Price ($/kWh)'] *= (1 + df['Electricity Annual Change']) ** effective_growth_years(df, config, 'electricity')
    df['Natural Gas Price ($/kWh)'] *= (1 + config['natural_gas_annual_growth']) ** effective_growth_years(df, config, 'natural_gas')
    df['Hydrogen Price ($/kWh)'] *= (1 + config['hydrogen_annual_change']) ** effective_growth_years(df, config, 'hydrogen')
    df['Biomass Price ($/kWh)'] *= (1 + df['Biomass Annual Change']) ** effective_growth_years(df, config, 'biomass')
    df['Carbon Credit Cap (tonnes CO2/year)'] *= (1 + config['carbon_credit_cap_annual_decline']) ** effective_growth_years(df, config, 'carbon_credit_cap')
    
    return df

def apply_gradual_adjustments(df, config):
    for adjustment in config.get('gradual_adjustments', []):
        start_date = pd.to_datetime(adjustment['start_date'])
        end_date = pd.to_datetime(adjustment['end_date'])
        variable = adjustment['variable']
        change_per_period = adjustment['change_per_period']
        frequency = adjustment['frequency']
        
        # Generate a date range for the adjustments
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Apply adjustments for each date in the range
        for date in dates:
            df.loc[df.index >= date, variable] *= (1 + change_per_period)
    return df

def calculate_renewable_share(df, config):
    start_year = config['renewable_share']['start_year']
    end_year = config['renewable_share']['end_year']
    start_share = config['renewable_share']['start_share']
    end_share = config['renewable_share']['end_share']
    
    total_years = end_year - start_year
    share_increment = (end_share - start_share) / total_years
    
    df['Renewable Share (%)'] = start_share + share_increment * (df['Year'] - start_year)
    df['Renewable Share (%)'] = df['Renewable Share (%)'].clip(lower=start_share, upper=end_share)
    return df


def incorporate_seasonality(df, config):
    winter_months = config['seasonality']['winter_months']
    summer_months = config['seasonality']['summer_months']
    
    df['Seasonal Factor'] = 1.0
    df.loc[df['Month'].isin(winter_months), 'Seasonal Factor'] = 1 + config['seasonality']['winter_increase']
    df.loc[df['Month'].isin(summer_months), 'Seasonal Factor'] = 1 - config['seasonality']['summer_decrease']
    
    df['Electricity Price ($/kWh)'] *= df['Seasonal Factor']
    df['Natural Gas Price ($/kWh)'] *= df['Seasonal Factor']
    
    df.drop(columns=['Seasonal Factor'], inplace=True)
    return df


def introduce_volatility(df, config):
    np.random.seed(config['random_seed'])
    
    # Base Electricity Volatility
    base_volatility = config['electricity_volatility_base']
    
    # Calculate volatility increase due to renewable share
    max_volatility_increase = config['renewable_volatility']['max_volatility_increase']
    df['Electricity Volatility'] = base_volatility + max_volatility_increase * (
        (df['Renewable Share (%)'] - config['renewable_share']['start_share']) /
        (config['renewable_share']['end_share'] - config['renewable_share']['start_share'])
    )
    
    # Technological improvements after some year
    tech_improvement_start_year = config['renewable_volatility']['tech_improvement_start_year']
    tech_volatility_reduction_rate = config['renewable_volatility']['tech_volatility_reduction_rate']
    tech_improvement_mask = df['Year'] >= tech_improvement_start_year
    df.loc[tech_improvement_mask, 'Electricity Volatility'] -= tech_volatility_reduction_rate * (df['Year'] - tech_improvement_start_year + 1)
    
    # Ensure Volatility Stays Within Bounds
    df['Electricity Volatility'] = df['Electricity Volatility'].clip(lower=0.01)
    
    # Apply Volatility to Electricity Prices
    df['Electricity Price ($/kWh)'] *= np.random.lognormal(mean=0, sigma=df['Electricity Volatility'], size=len(df))
    
    # Natural Gas Price Volatility
    natural_gas_volatility = config['natural_gas_volatility']
    df['Natural Gas Price ($/kWh)'] *= np.random.lognormal(mean=0, sigma=natural_gas_volatility, size=len(df))
    
    # Biomass Price Volatility
    biomass_volatility = config['biomass_volatility']
    df['Biomass Price ($/kWh)'] *= np.random.lognormal(mean=0, sigma=biomass_volatility, size=len(df))
    
    # Hydrogen Price Volatility
    hydrogen_volatility = config['hydrogen_volatility']
    df['Hydrogen Price ($/kWh)'] *= np.random.lognormal(mean=0, sigma=hydrogen_volatility, size=len(df))
    
    return df


def adjust_correlations(df, config):
    # Hydrogen and Electricity Correlation based on Renewable Share
    df['Hydrogen Correlation'] = config['hydrogen_correlation_base'] + \
        (df['Renewable Share (%)'] - config['renewable_share']['start_share']) / \
        (config['renewable_share']['end_share'] - config['renewable_share']['start_share']) * \
        (config['hydrogen_correlation_max'] - config['hydrogen_correlation_base'])
    df['Hydrogen Correlation'] = df['Hydrogen Correlation'].clip(upper=config['hydrogen_correlation_max'])

    # Natural Gas and Electricity Correlation
    df['Natural Gas Correlation'] = config['natural_gas_correlation_base'] - \
        (df['Renewable Share (%)'] - config['renewable_share']['start_share']) / \
        (config['renewable_share']['end_share'] - config['renewable_share']['start_share']) * \
        (config['natural_gas_correlation_base'] - config['natural_gas_correlation_min'])
    df['Natural Gas Correlation'] = df['Natural Gas Correlation'].clip(lower=config['natural_gas_correlation_min'])

    # Updated Biomass and Natural Gas Correlation
    base_corr = config['biomass_natural_gas_correlation_base']
    decay = config.get('biomass_natural_gas_correlation_decay', 0.5)
    start_share = config['renewable_share']['start_share']
    end_share = config['renewable_share']['end_share']
    
    renewable_factor = (df['Renewable Share (%)'] - start_share) / (end_share - start_share)
    df['Biomass NG Correlation'] = base_corr * (1 - decay * renewable_factor)
    df['Biomass NG Correlation'] = df['Biomass NG Correlation'].clip(lower=config.get('biomass_natural_gas_correlation_min', 0.05))
    df['Biomass Price ($/kWh)'] = (df['Biomass Price ($/kWh)'] * (1 - df['Biomass NG Correlation'])) + \
                                   (df['Natural Gas Price ($/kWh)'] * df['Biomass NG Correlation'])
    return df


def implement_events(df, config):
    for event in config['events']:
        start_date = pd.to_datetime(event['start_date'])
        end_date = pd.to_datetime(event['end_date'])
        impacts = event['impacts']
        event_mask = (df.index >= start_date) & (df.index <= end_date)
        for variable, impact in impacts.items():
            if isinstance(impact, dict):
                # Gradual change
                change_per_period = impact['change_per_period']
                frequency = impact['frequency']
                dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
                for date in dates:
                    mask = (df.index >= date) & event_mask
                    df.loc[mask, variable] *= (1 + change_per_period)
            else:
                # Immediate adjustment
                df.loc[event_mask, variable] *= impact
    return df


def apply_random_demand_shocks(df, config):
    np.random.seed(config['random_seed'])
    years = df['Year'].unique()
    for year in years:
        months = np.random.choice(range(1, 13), config['demand_shocks']['frequency_per_year'], replace=False)
        for month in months:
            shock = np.random.uniform(*config['demand_shocks']['impact_range'])
            shock_mask = (df['Year'] == year) & (df['Month'] == month)
            df.loc[shock_mask, 'Electricity Price ($/kWh)'] *= shock
    return df


def model_exchange_rate(df, config):
    np.random.seed(config['random_seed'])
    df['Exchange Rate Index'] = 1.0
    exchange_rate_volatility = config['exchange_rate_volatility']
    df['Exchange Rate Index'] *= np.random.lognormal(mean=0, sigma=exchange_rate_volatility, size=len(df))
    df['Exchange Rate Index'] *= (1 + config['exchange_rate_trend']) ** df['Years Since Start']
    return df


def adjust_for_exchange_rate(df, config):
    df['Natural Gas Price ($/kWh)'] *= df['Exchange Rate Index']
    df['Biomass Price ($/kWh)'] *= df['Exchange Rate Index']
    return df


def apply_economic_cycles(df, config):
    for cycle in config['economic_cycles']:
        start_date = pd.to_datetime(cycle['start_date'])
        end_date = pd.to_datetime(cycle['end_date'])
        impact = cycle['impact']
        cycle_mask = (df.index >= start_date) & (df.index <= end_date)
        for variable, multiplier in impact.items():
            df.loc[cycle_mask, variable] *= multiplier
    return df


def introduce_market_speculation(df, config):
    np.random.seed(config['random_seed'])
    for commodity in config['speculative_commodities']:
        spike_prob = config['speculation_spike_probability']
        spike_multiplier = config['speculation_spike_multiplier']
        random_values = np.random.rand(len(df))
        spike_mask = random_values < spike_prob
        df.loc[spike_mask, commodity] *= np.random.uniform(1, spike_multiplier)
    return df


def adjust_for_ccs_and_policies(df, config):
    ccs_start_year = config['ccs_start_year']
    ccs_max_reduction = config['ccs_max_reduction']
    ccs_annual_increase = config['ccs_annual_increase']
    
    df['CCS Reduction Factor'] = 1.0
    ccs_years = df['Year'] >= ccs_start_year
    df.loc[ccs_years, 'CCS Reduction Factor'] = 1 - (ccs_annual_increase * (df['Year'] - ccs_start_year + 1))
    df['CCS Reduction Factor'] = df['CCS Reduction Factor'].clip(lower=1 - ccs_max_reduction)
    
    df['Effective Carbon Credit Cap'] = df['Carbon Credit Cap (tonnes CO2/year)'] * df['CCS Reduction Factor']
    
    # Carbon Tax
    carbon_tax_start_year = config['carbon_tax_start_year']
    df['Carbon Tax ($/tonne CO2)'] = 0.0
    df.loc[df['Year'] >= carbon_tax_start_year, 'Carbon Tax ($/tonne CO2)'] = \
        config['carbon_tax_start_rate'] + config['carbon_tax_increase'] * \
        ((df['Year'] - carbon_tax_start_year) // config['carbon_tax_increase_interval'])
    
    return df


def calculate_emissions(df, config):
    df['Electricity Emissions (kg CO2)'] = df['Electricity Price ($/kWh)'] * df['Grid Carbon Intensity (kg CO2/kWh)']
    df['Natural Gas Emissions (kg CO2)'] = df['Natural Gas Price ($/kWh)'] * df['Natural Gas Carbon Intensity (kg CO2/kWh)']
    df['Hydrogen Emissions (kg CO2)'] = df['Hydrogen Price ($/kWh)'] * df['Hydrogen Carbon Intensity (kg CO2/kWh)']
    df['Biomass Emissions (kg CO2)'] = df['Biomass Price ($/kWh)'] * df['Biomass Carbon Intensity (kg CO2/kWh)']
    return df


def apply_inflation_adjustment(df, inflation_rates):
    inflation_df = pd.DataFrame(list(inflation_rates.items()), columns=['Year', 'Inflation Rate'])
    inflation_df['Year'] = inflation_df['Year'].astype(int)
    inflation_df.sort_values('Year', inplace=True)

    inflation_df['Cumulative Inflation'] = (1 + inflation_df['Inflation Rate']).cumprod()

    df = df.reset_index()
    df = df.merge(inflation_df[['Year', 'Cumulative Inflation']], on='Year', how='left')

    # -- IMPORTANT: fill any missing inflation with 1.0
    df['Cumulative Inflation'] = df['Cumulative Inflation'].fillna(1.0)

    df.set_index('index', inplace=True)
    df.index.name = None
    return df


def adjust_for_inflation(df, price_columns):
    """
    Divides out 'Cumulative Inflation' from each column in price_columns,
    giving real $ in today's dollars.
    """
    for col in price_columns:
        if col in df.columns:
            df[col] /= df['Cumulative Inflation']
    return df


def calculate_grid_carbon_intensity(df, config):
    base_intensity = config['grid_carbon_intensity']['base_intensity']
    target_intensity = config['grid_carbon_intensity']['target_intensity']
    
    df['Grid Carbon Intensity (kg CO2/kWh)'] = base_intensity - (
        (df['Renewable Share (%)'] - config['renewable_share']['start_share']) /
        (config['renewable_share']['end_share'] - config['renewable_share']['start_share'])
    ) * (base_intensity - target_intensity)
    
    variation = config['grid_carbon_intensity'].get('variation', {})
    if variation.get('daily_cycle', False):
        peak_hours = variation.get('peak_hours', [])
        peak_multiplier = variation.get('peak_multiplier', 1.0)
        off_peak_multiplier = variation.get('off_peak_multiplier', 1.0)
        df['Carbon Intensity Multiplier'] = off_peak_multiplier
        df.loc[df['Hour'].isin(peak_hours), 'Carbon Intensity Multiplier'] = peak_multiplier
        df['Grid Carbon Intensity (kg CO2/kWh)'] *= df['Carbon Intensity Multiplier']
        df.drop(columns=['Carbon Intensity Multiplier'], inplace=True)
    
    df['Grid Carbon Intensity (kg CO2/kWh)'] = df['Grid Carbon Intensity (kg CO2/kWh)'].clip(lower=0)
    return df


def calculate_natural_gas_carbon_intensity(df, config):
    base_intensity = config['natural_gas_carbon_intensity']['base_intensity']
    peak_intensity = config['natural_gas_carbon_intensity']['peak_intensity']
    start_year = config['natural_gas_carbon_intensity']['transition_start_year']
    end_year = config['natural_gas_carbon_intensity']['transition_end_year']
    
    total_years = end_year - start_year
    intensity_increment = (peak_intensity - base_intensity) / total_years
    df['Natural Gas Carbon Intensity (kg CO2/kWh)'] = base_intensity
    transition_mask = (df['Year'] >= start_year) & (df['Year'] <= end_year)
    df.loc[transition_mask, 'Natural Gas Carbon Intensity (kg CO2/kWh)'] = base_intensity + \
        intensity_increment * (df['Year'] - start_year)
    df.loc[df['Year'] > end_year, 'Natural Gas Carbon Intensity (kg CO2/kWh)'] = peak_intensity
    return df


def calculate_hydrogen_carbon_intensity(df, config):
    base_intensity = config['hydrogen_carbon_intensity']['base_intensity']
    target_intensity = config['hydrogen_carbon_intensity']['target_intensity']
    start_year = config['hydrogen_carbon_intensity']['transition_start_year']
    end_year = config['hydrogen_carbon_intensity']['transition_end_year']
    
    total_years = end_year - start_year
    intensity_decrement = (base_intensity - target_intensity) / total_years
    df['Hydrogen Carbon Intensity (kg CO2/kWh)'] = base_intensity
    transition_mask = (df['Year'] >= start_year) & (df['Year'] <= end_year)
    df.loc[transition_mask, 'Hydrogen Carbon Intensity (kg CO2/kWh)'] = base_intensity - \
        intensity_decrement * (df['Year'] - start_year)
    df.loc[df['Year'] > end_year, 'Hydrogen Carbon Intensity (kg CO2/kWh)'] = target_intensity
    return df


def calculate_biomass_carbon_intensity(df, config):
    intensity = config['biomass_carbon_intensity']['fixed_intensity']
    df['Biomass Carbon Intensity (kg CO2/kWh)'] = intensity
    return df


def generate_feedstock_inputs(df, config):
    ng_relative_change = (df['Natural Gas Price ($/kWh)'] / config['natural_gas_base_price']) - 1
    df['BPA Price ($/tonne)'] = config['bpa_base_price'] * (1 + config['bpa_correlation_with_ng'] * ng_relative_change)
    df['ECH Price ($/tonne)'] = config['ech_base_price'] * (1 + config['ech_correlation_with_ng'] * ng_relative_change)
    df['Feedstock Price ($/tonne)'] = 0.6 * df['BPA Price ($/tonne)'] + 0.4 * df['ECH Price ($/tonne)']
    return df


def generate_fixed_cost_inputs(df, config):
    fixed = config['fixed_costs']
    total_fixed = (
        fixed['labour_cost_per_year'] +
        fixed['overhead_cost_per_year'] +
        fixed['maintenance_cost_per_year'] +
        fixed['depreciation_per_year'] +
        fixed['environmental_compliance_cost_per_year']
    )
    production = config.get('production_tonnage', 100000)

    growth_rate = config.get('fixed_cost_growth', 0.02)
    df['Fixed Cost ($/tonne)'] = (total_fixed / production) * ((1 + growth_rate) ** (df['Year'] - config['start_year']))
    return df


def generate_dynamic_resin_price(df, config):
    # Determine data frequency
    sample_year = df[df['Year'] == df['Year'].iloc[0]]
    rows_per_year = len(sample_year)
    dt = 1/8760.0 if rows_per_year > 12 else 1/12.0

    base_price = config['selling_price_per_tonne']
    mu = config.get('resin_mu', 0.02)
    sigma = config.get('resin_sigma', 0.05)
    
    prices = [base_price]
    np.random.seed(config['random_seed'])
    for i in range(1, len(df)):
        z = np.random.normal()
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * z)
        prices.append(new_price)
    df['Resin Selling Price ($/tonne)'] = prices

    # Apply event multipliers, if any
    if 'resin_events' in config:
        for event in config['resin_events']:
            start_year = event.get('start_year')
            end_year = event.get('end_year', start_year)
            multiplier = event.get('price_multiplier', 1.0)
            event_mask = (df['Year'] >= start_year) & (df['Year'] <= end_year)
            df.loc[event_mask, 'Resin Selling Price ($/tonne)'] *= multiplier

    return df


def calculate_input_margin(df, config):
    df['Input Margin ($/tonne)'] = df['Resin Selling Price ($/tonne)'] - (
        df['Feedstock Price ($/tonne)'] + df['Fixed Cost ($/tonne)']
    )
    return df


def save_datasets(df, config):
    df.to_csv("data/"+config['hourly_dataset_filename'])
    print(f"Hourly dataset generation complete. Saved as '{config['hourly_dataset_filename']}'.")

    monthly_df = df.resample('M').mean()
    monthly_df['Carbon Credit Cap (tonnes CO2/year)'] = df['Carbon Credit Cap (tonnes CO2/year)'].resample('M').first()
    monthly_df['Effective Carbon Credit Cap'] = df['Effective Carbon Credit Cap'].resample('M').first()
    monthly_df['Year'] = monthly_df.index.year
    monthly_df['Month'] = monthly_df.index.month
    
    monthly_df.to_csv("data/"+config['monthly_dataset_filename'], index=False)
    print(f"Monthly average dataset generation complete. Saved as '{config['monthly_dataset_filename']}'.")


def generate_dataset(config):
    df = initialize_dataframe(config['start_date'], config['end_date'])
    
    # Base prices and conversions:
    df['Electricity Price ($/kWh)'] = config['electricity_base_price']
    df['Natural Gas Price ($/kWh)'] = config['natural_gas_base_price']
    # Convert hydrogen base price per kg into $/kWh using energy content
    df['Hydrogen Price ($/kWh)'] = config['hydrogen_base_price_per_kg'] / config['hydrogen_energy_content']
    df['Biomass Price ($/kWh)'] = config['biomass_base_price_per_tonne'] / config['biomass_energy_content_per_tonne']
    df['Carbon Credit Cap (tonnes CO2/year)'] = config['carbon_credit_cap_base']
    
    # Carbon Credit Price: linear interpolation from base to target by projection year
    years = np.array([config['start_year'], config['carbon_credit_price_projection_year']])
    prices = np.array([config['carbon_credit_base_price'], config['carbon_credit_price_target']])
    poly = np.polyfit(years, prices, 1)
    df['Carbon Credit Price ($/tonne CO2)'] = np.polyval(poly, df['Year'])
    
    # Add lognormal volatility to carbon credit price
    np.random.seed(config['random_seed'])
    df['Carbon Credit Price ($/tonne CO2)'] *= np.random.lognormal(mean=0, sigma=config['carbon_credit_volatility'], size=len(df))
    
    # Apply growth adjustments (with stabilization)
    df = apply_annual_growth(df, config)
    
    # Continue with your other steps:
    df = calculate_renewable_share(df, config)
    df = calculate_grid_carbon_intensity(df, config)
    df = calculate_natural_gas_carbon_intensity(df, config)
    df = calculate_hydrogen_carbon_intensity(df, config)
    df = calculate_biomass_carbon_intensity(df, config)
    
    df = apply_gradual_adjustments(df, config)
    df = model_exchange_rate(df, config)
    df = adjust_for_exchange_rate(df, config)
    df = incorporate_seasonality(df, config)
    df = introduce_volatility(df, config)
    df = adjust_correlations(df, config)
    df = implement_events(df, config)
    df = apply_random_demand_shocks(df, config)
    df = apply_economic_cycles(df, config)
    df = introduce_market_speculation(df, config)
    df = adjust_for_ccs_and_policies(df, config)
    df = calculate_emissions(df, config)
    
    df = apply_inflation_adjustment(df, config['inflation_rates'])
    price_columns = [
        'Electricity Price ($/kWh)',
        'Natural Gas Price ($/kWh)',
        'Hydrogen Price ($/kWh)',
        'Biomass Price ($/kWh)',
        'Carbon Credit Price ($/tonne CO2)',
        'Carbon Tax ($/tonne CO2)'
    ]
    df = adjust_for_inflation(df, price_columns)
    
    # Additional inputs
    df = generate_feedstock_inputs(df, config)
    df = generate_fixed_cost_inputs(df, config)
    df = generate_dynamic_resin_price(df, config)
    df = calculate_input_margin(df, config)
    
    # Apply discount factor for NPV calculations
    df = apply_real_discount_factor(df, config)
    
    save_datasets(df, config)
    return df


# NEW: a function to discount each row's real values
def apply_real_discount_factor(df, config):
    """
    Multiply each row's cost/revenue columns by 1/(1 + real_rate)^(Years Since Start).
    Adds new columns with suffix ' PV' for present value in real terms.
    """
    real_discount_rate = config.get('real_discount_rate', 0.05)
    
    if 'Years Since Start' not in df.columns:
        df['Years Since Start'] = df['Year'] - config['start_year']
    
    df['Discount Factor'] = 1.0 / ((1.0 + real_discount_rate) ** df['Years Since Start'])

    # For example, discount the feedstock cost, fixed cost, resin price, margin, etc.
    columns_to_discount = [
        'Fixed Cost ($/tonne)',
        'Resin Selling Price ($/tonne)',
        'Input Margin ($/tonne)'
        # Add more if you want discounted electricity or others
    ]
    for col in columns_to_discount:
        if col in df.columns:
            df[f'{col} PV'] = df[col] * df['Discount Factor']

    return df


def calculate_capex_costs(df, config):
    capex_config = config['capex']
    years = df['Year'].unique()
    capex_df = pd.DataFrame({'Year': years})

    for tech_name, tech_config in capex_config.items():
        capex_df = calculate_technology_capex(capex_df, tech_name, tech_config, config)

    # Adjust for inflation => real
    capex_df = capex_df.merge(df[['Year', 'Cumulative Inflation']].drop_duplicates(), on='Year', how='left')
    capex_columns = [col for col in capex_df.columns if 'CAPEX' in col]
    for col in capex_columns:
        capex_df[col] /= capex_df['Cumulative Inflation']
    capex_df.drop(columns=['Cumulative Inflation'], inplace=True)

    # NEW: discount the CAPEX if you want present value
    real_discount_rate = config.get('real_discount_rate', 0.05)
    capex_df['Years Since Start'] = capex_df['Year'] - config['start_year']
    capex_df['Discount Factor'] = 1.0 / ((1.0 + real_discount_rate) ** capex_df['Years Since Start'])
    
    for col in capex_columns:
        capex_df[f'{col} PV'] = capex_df[col] * capex_df['Discount Factor']
    
    capex_df.drop(columns=['Years Since Start', 'Discount Factor'], inplace=True)
    return capex_df


def calculate_technology_capex(capex_df, tech_name, tech_config, config):
    years = capex_df['Year']
    base_cost = tech_config['base_cost']
    learning_rate = tech_config['learning_rate']
    initial_capacity = tech_config['initial_installed_capacity']
    annual_growth_rate = tech_config['annual_capacity_growth_rate']
    events = tech_config.get('events', [])
    capex_column_name = f"{tech_name} CAPEX ({tech_config['cost_unit']})"
    capex_list = []
    cumulative_capacity = initial_capacity
    cost = base_cost
    
    for year in years:
        # Process events if any occur in this year
        for event in events.copy():
            if 'start_year' in event and 'end_year' in event:
                if event['start_year'] <= year <= event['end_year']:
                    change_per_period = event.get('change_per_period', 0)
                    cost *= (1 + change_per_period)
            elif 'start_year' in event:
                if year >= event['start_year']:
                    cost *= (1 - event.get('cost_reduction', 0))
                    events.remove(event)
                    break

        # Apply learning curve
        if cumulative_capacity > 0:
            capacity_factor = cumulative_capacity / initial_capacity
            learning_adjustment = capacity_factor ** (np.log2(1 - learning_rate))
            # Capex inflation
            escalation_factor = (1 + config.get('capex_inflation_rate', 0.02)) ** (year - config['start_year'])
            cost = base_cost * learning_adjustment * escalation_factor

        capex_list.append(cost)

        # next year capacity
        cumulative_capacity *= (1 + annual_growth_rate)

    capex_df[capex_column_name] = capex_list
    return capex_df


def save_capex_dataset(capex_df, config):
    capex_filename = config.get('capex_dataset_filename', 'capex_costs_over_time.csv')
    capex_df.to_csv("data/"+capex_filename, index=False)
    print(f"CAPEX dataset generation complete. Saved as '{capex_filename}'.")


def plot_volatility_and_renewable_share(df):
    import matplotlib.pyplot as plt
    volatility_df = df[['Year', 'Electricity Volatility', 'Renewable Share (%)']].drop_duplicates()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Electricity Volatility', color='tab:blue')
    ax1.plot(volatility_df['Year'], volatility_df['Electricity Volatility'], color='tab:blue', marker='o', label='Electricity Volatility')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Renewable Share (%)', color='tab:green')
    ax2.plot(volatility_df['Year'], volatility_df['Renewable Share (%)'], color='tab:green', marker='x', label='Renewable Share (%)')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    fig.tight_layout()
    plt.title('Electricity Volatility and Renewable Share Over Time')
    plt.show()


def main():
    config = load_config()
    df = generate_dataset(config)
    plot_volatility_and_renewable_share(df)
    
    # CAPEX calculations remain unchanged.
    capex_df = calculate_capex_costs(df, config)
    save_capex_dataset(capex_df, config)

if __name__ == "__main__":
    main()