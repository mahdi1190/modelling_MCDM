from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyomo.environ import value
from matplotlib.lines import Line2D

# Append the configuration path if necessary
# If 'solver_options.py' is in a different directory, adjust the path accordingly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

# Import the solver options (ensure 'solver_options.py' is accessible)
from solver_options import get_solver

current_dir = os.path.dirname(__file__)

# Construct paths to the data files
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'optimized_demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))

# Read the Excel files
demands = pd.read_excel(demands_path)
markets_monthly = pd.read_excel(markets_monthly_path)
markets = pd.read_excel(markets_path)

# Extract demand and market data
electricity_demand = demands["purchased_elec"].to_numpy()  # kWh
nat_gas = demands["nat_gas"].to_numpy()  # m3
credits_sold = demands["credits_held"].to_numpy()
credits_purchased = demands["credits_needed"].to_numpy()

electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

NG_market = markets["nat_gas"].to_numpy()
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

heat_market_sold = markets["nat_gas_sold"].to_numpy()

H2_market = markets["hydrogen"].to_numpy()
BM_market = markets["biomass"].to_numpy()

CHP_capacity = 2500

total_times = 180  
total_hours = 24 * total_times
no_contract = 8
time_limit = 30
bound_duration = 180

risk_appetite_threshold = 1000

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # Sets
    total_time = total_times
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    MONTHS = np.arange(total_time)
    model.MONTHS = Set(initialize=MONTHS)

    no_intervals = 4
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)
    model.INTERVALS = Set(initialize=INTERVALS)

    model.CONTRACTS = Set(initialize=range(no_contract))
    model.ContractTypes = Set(initialize=['Fixed', 'Indexed', 'TakeOrPay', 'Fixed_Plus'])

    # Parameters
    # Define min and max prices based on NG_market_monthly data
    min_price = min(NG_market_monthly)  # Adjusted for safety margin
    max_price = max(NG_market_monthly)  # Adjusted for safety margin
    M = 1e6  # Large constant for big-M method

    # Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)

    # Binary and continuous variables for contracts
    model.ContractStart = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractActive = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractType = Var(model.CONTRACTS, model.ContractTypes, within=Binary)
    model.ContractAmount = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.ContractPrice = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(min_price, max_price))
    model.ContractStartPrice = Var(model.CONTRACTS, within=NonNegativeReals, bounds=(min_price, max_price))
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(0, bound_duration))

    # New variable for Fixed_Plus contracts
    model.ContractInitialAmount = Var(model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))

    model.risk_score = Var(within=NonNegativeReals, bounds=(0, risk_appetite_threshold))

    # Auxiliary variables for linearization
    model.ContractAmountActive = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.TotalCost = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals)

    # Auxiliary variables for products of variables
    model.FixedPlusAmount = Var(model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.MinAmountActive = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals)
    model.MaxAmountActive = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals)

    # Constraints

    # Duration constraint for each contract
    def contract_duration_rule(model, c):
        return sum(model.ContractActive[m, c] for m in model.MONTHS) == model.ContractDuration[c]
    model.ContractDurationCon = Constraint(model.CONTRACTS, rule=contract_duration_rule)

    # Ensure minimum duration for each contract after it starts
    def min_contract_duration_rule(model, c):
        return sum(model.ContractActive[m, c] for m in model.MONTHS) >= 3 * sum(model.ContractStart[m, c] for m in model.MONTHS)
    model.MinContractDurationCon = Constraint(model.CONTRACTS, rule=min_contract_duration_rule)

    # Ensure only one contract type is selected per contract
    def contract_type_rule(model, c):
        return sum(model.ContractType[c, t] for t in model.ContractTypes) == 1
    model.ContractTypeConstraint = Constraint(model.CONTRACTS, rule=contract_type_rule)

    # Linking contract start to active months
    def contract_active_rule(model, m, c):
        if m == 0:
            return model.ContractActive[m, c] == model.ContractStart[m, c]
        else:
            return model.ContractActive[m, c] >= model.ContractActive[m - 1, c] + model.ContractStart[m, c] - 1
    model.ContractActiveCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_active_rule)

    # Ensure only one contract start per contract
    def contract_single_start_rule(model, c):
        return sum(model.ContractStart[m, c] for m in model.MONTHS) <= 1
    model.ContractSingleStartCon = Constraint(model.CONTRACTS, rule=contract_single_start_rule)

    # Set start price based on contract's start month
    def set_contract_start_price_rule(model, c):
        return model.ContractStartPrice[c] == sum(NG_market_monthly[m] * model.ContractStart[m, c] for m in model.MONTHS)
    model.SetContractStartPrice = Constraint(model.CONTRACTS, rule=set_contract_start_price_rule)

    # Set initial contract amount based on the start month for Fixed_Plus contracts
    def set_contract_initial_amount_rule(model, c):
        return model.ContractInitialAmount[c] == sum(model.ContractAmount[m, c] * model.ContractStart[m, c] for m in model.MONTHS)
    model.SetContractInitialAmount = Constraint(model.CONTRACTS, rule=set_contract_initial_amount_rule)

    # Linearization for FixedPlusAmount = ContractType['Fixed_Plus'] * ContractInitialAmount
    def fixed_plus_amount_rule1(model, c):
        return model.FixedPlusAmount[c] <= model.ContractInitialAmount[c]
    def fixed_plus_amount_rule2(model, c):
        return model.FixedPlusAmount[c] <= M * model.ContractType[c, 'Fixed_Plus']
    def fixed_plus_amount_rule3(model, c):
        return model.FixedPlusAmount[c] >= model.ContractInitialAmount[c] - M * (1 - model.ContractType[c, 'Fixed_Plus'])
    model.FixedPlusAmountCon1 = Constraint(model.CONTRACTS, rule=fixed_plus_amount_rule1)
    model.FixedPlusAmountCon2 = Constraint(model.CONTRACTS, rule=fixed_plus_amount_rule2)
    model.FixedPlusAmountCon3 = Constraint(model.CONTRACTS, rule=fixed_plus_amount_rule3)

    # Contract price setting rule
    def contract_price_setting_rule(model, m, c):
        # Fixed price contracts
        fixed_price = model.ContractType[c, 'Fixed'] * model.ContractStartPrice[c] * 0.8
        # Indexed contracts
        indexed_price = model.ContractType[c, 'Indexed'] * NG_market_monthly[m]
        # TakeOrPay contracts
        take_or_pay_price = model.ContractType[c, 'TakeOrPay'] * model.ContractStartPrice[c] * 0.3
        # Fixed_Plus contracts
        fixed_plus_price = model.ContractType[c, 'Fixed_Plus'] * model.ContractStartPrice[c] * 0.5  # Adjusted pricing
        return model.ContractPrice[m, c] == fixed_price + indexed_price + take_or_pay_price + fixed_plus_price
    model.ContractPriceSetting = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_price_setting_rule)

    # Minimum contract fulfillment amount
    def min_fulfillment_rule(model, m, c):
        min_amount = (
            1000 * model.ContractType[c, 'TakeOrPay'] +
            400 * model.ContractType[c, 'Fixed'] +
            400 * model.ContractType[c, 'Indexed'] +
            model.FixedPlusAmount[c]
        )
        # Linearization for MinAmountActive = ContractActive * min_amount
        return model.MinAmountActive[m, c] >= min_amount - M * (1 - model.ContractActive[m, c])
    model.MinFulfillmentCon = Constraint(model.MONTHS, model.CONTRACTS, rule=min_fulfillment_rule)

    def min_amount_active_upper_rule(model, m, c):
        min_amount = (
            1000 * model.ContractType[c, 'TakeOrPay'] +
            400 * model.ContractType[c, 'Fixed'] +
            400 * model.ContractType[c, 'Indexed'] +
            model.FixedPlusAmount[c]
        )
        return model.MinAmountActive[m, c] <= min_amount
    model.MinAmountActiveUpperCon = Constraint(model.MONTHS, model.CONTRACTS, rule=min_amount_active_upper_rule)

    def min_amount_active_zero_rule(model, m, c):
        return model.MinAmountActive[m, c] <= M * model.ContractActive[m, c]
    model.MinAmountActiveZeroCon = Constraint(model.MONTHS, model.CONTRACTS, rule=min_amount_active_zero_rule)

    # Ensure that ContractAmount >= MinAmountActive
    def contract_amount_min_rule(model, m, c):
        return model.ContractAmount[m, c] >= model.MinAmountActive[m, c]
    model.ContractAmountMinCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_amount_min_rule)

    # Maximum contract fulfillment amount
    def max_fulfillment_rule(model, m, c):
        max_amount = (
            800 * model.ContractType[c, 'TakeOrPay'] +
            500 * model.ContractType[c, 'Fixed'] +
            2000 * model.ContractType[c, 'Indexed'] +
            model.FixedPlusAmount[c]
        )
        # Linearization for MaxAmountActive = ContractActive * max_amount
        return model.MaxAmountActive[m, c] <= max_amount + M * (1 - model.ContractActive[m, c])
    model.MaxFulfillmentCon = Constraint(model.MONTHS, model.CONTRACTS, rule=max_fulfillment_rule)

    def max_amount_active_lower_rule(model, m, c):
        return model.MaxAmountActive[m, c] >= 0
    model.MaxAmountActiveLowerCon = Constraint(model.MONTHS, model.CONTRACTS, rule=max_amount_active_lower_rule)

    def max_amount_active_zero_rule(model, m, c):
        return model.MaxAmountActive[m, c] <= M * model.ContractActive[m, c]
    model.MaxAmountActiveZeroCon = Constraint(model.MONTHS, model.CONTRACTS, rule=max_amount_active_zero_rule)

    # Ensure that ContractAmount <= MaxAmountActive
    def contract_amount_max_rule(model, m, c):
        return model.ContractAmount[m, c] <= model.MaxAmountActive[m, c]
    model.ContractAmountMaxCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_amount_max_rule)

    # Enforce fixed amount for Fixed_Plus contracts across active months
    def fixed_plus_contract_amount_rule_upper(model, m, c):
        return model.ContractAmount[m, c] - model.ContractInitialAmount[c] <= M * (1 - model.ContractType[c, 'Fixed_Plus'])
    model.FixedPlusContractAmountUpperCon = Constraint(model.MONTHS, model.CONTRACTS, rule=fixed_plus_contract_amount_rule_upper)

    def fixed_plus_contract_amount_rule_lower(model, m, c):
        return model.ContractAmount[m, c] - model.ContractInitialAmount[c] >= -M * (1 - model.ContractType[c, 'Fixed_Plus'])
    model.FixedPlusContractAmountLowerCon = Constraint(model.MONTHS, model.CONTRACTS, rule=fixed_plus_contract_amount_rule_lower)

    # Ensure demand fulfillment for natural gas
    def demand_fulfillment_rule(model, m):
        return sum(model.ContractAmountActive[m, c] for c in model.CONTRACTS) >= nat_gas[m]
    model.demand_fulfillment_con = Constraint(model.MONTHS, rule=demand_fulfillment_rule)

    # Linearization of ContractAmountActive = ContractAmount * ContractActive
    def contract_amount_active_rule1(model, m, c):
        return model.ContractAmountActive[m, c] <= model.ContractAmount[m, c]
    model.ContractAmountActiveUpper = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_amount_active_rule1)

    def contract_amount_active_rule2(model, m, c):
        return model.ContractAmountActive[m, c] <= model.ContractActive[m, c] * 2000
    model.ContractAmountActiveBinary = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_amount_active_rule2)

    def contract_amount_active_rule3(model, m, c):
        return model.ContractAmountActive[m, c] >= model.ContractAmount[m, c] - (1 - model.ContractActive[m, c]) * 2000
    model.ContractAmountActiveMin = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_amount_active_rule3)

    # Risk score calculation
    def risk_score_rule(model):
        return model.risk_score == sum(
            (1 * model.ContractType[c, 'Fixed'] +
             3 * model.ContractType[c, 'Indexed'] +
             2 * model.ContractType[c, 'TakeOrPay'] +
             1.5 * model.ContractType[c, 'Fixed_Plus']) *  # Adjusted risk score
            model.ContractDuration[c]
            for c in model.CONTRACTS
        )
    model.RiskScoreCon = Constraint(rule=risk_score_rule)

    # McCormick Envelopes for TotalCost = ContractAmountActive * ContractPrice
    model.McCormickConstraints = ConstraintList()
    for m in model.MONTHS:
        for c in model.CONTRACTS:
            x = model.ContractAmountActive[m, c]
            y = model.ContractPrice[m, c]
            z = model.TotalCost[m, c]

            xL = x.lb
            xU = x.ub
            yL = y.lb
            yU = y.ub

            # McCormick inequalities
            model.McCormickConstraints.add(z >= xL * y + x * yL - xL * yL)
            model.McCormickConstraints.add(z >= xU * y + x * yU - xU * yU)
            model.McCormickConstraints.add(z <= xU * y + x * yL - xU * yL)
            model.McCormickConstraints.add(z <= xL * y + x * yU - xL * yU)

    # Objective function
    def objective_rule(model):
        return sum(model.TotalCost[m, c] for m in model.MONTHS for c in model.CONTRACTS)
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Solver configuration
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.options['MIPGap'] = 0.01          # Tighter optimality gap
    solver.options['Threads'] = 0            # Use all available threads
    solver.options['MIPFocus'] = 1           # Focus on finding feasible solutions quickly
    solver.options['Presolve'] = 2           # Aggressive presolve
    solver.options['Heuristics'] = 0.5       # Increase heuristic search
    solver.options['Cuts'] = 2               # Aggressive cut generation
    solver.options['Symmetry'] = 2           # Aggressive symmetry detection
    solver.options['Method'] = 2             # Use Barrier method for LP relaxations

    # Solve the model
    results = solver.solve(model, tee=True, symbolic_solver_labels=False)

    return model



if __name__ == '__main__':
    model = pyomomodel()

    # Extract data from the model for plotting
    # Initialize dictionaries to store data
    start_months = {}
    durations = {}
    contract_types = {}
    contract_amounts = {}  # To store the total amount per contract
    contract_prices = {}   # To store the starting price per contract

    # For detailed monthly amounts
    monthly_data = []  # List to store data for the stacked area chart

    # Loop over all contracts
    for c in model.CONTRACTS:
        # Find the start month
        start_month = None
        for m in model.MONTHS:
            if value(model.ContractStart[m, c]) >= 0.99:
                start_month = m
                break
        if start_month is None:
            # Contract did not start, skip
            continue
        start_months[c] = start_month

        # Get duration
        duration = value(model.ContractDuration[c])
        durations[c] = duration

        # Get contract type
        contract_type = None
        for t in model.ContractTypes:
            if value(model.ContractType[c, t]) >= 0.99:
                contract_type = t
                break
        contract_types[c] = contract_type

        # Get total amount over the duration
        total_amount = sum(value(model.ContractAmount[m, c]) for m in model.MONTHS if value(model.ContractActive[m, c]) >= 0.99)
        contract_amounts[c] = total_amount

        # Get contract start price
        start_price = value(model.ContractStartPrice[c])
        contract_prices[c] = start_price

        # Collect monthly data for this contract
        for m in model.MONTHS:
            if value(model.ContractActive[m, c]) >= 0.99:
                amount = value(model.ContractAmount[m, c])
            else:
                amount = 0
            monthly_data.append({
                'Contract': c,
                'Month': m,
                'Amount': amount,
                'Type': contract_type,
            })

    # Create a DataFrame for easy handling
    contract_data = pd.DataFrame({
        'Contract': list(start_months.keys()),
        'Start Month': [start_months[c] for c in start_months],
        'Duration': [durations[c] for c in start_months],
        'Type': [contract_types[c] for c in start_months],
        'Total Amount': [contract_amounts[c] for c in start_months],
        'Start Price': [contract_prices[c] for c in start_months],
    })

    print(contract_data)

    # Optionally, save the data to a CSV file
    contract_data.to_csv('contract_data.csv', index=False)

    # Plotting the Contract Activation Over Time
    # Mapping contract types to colors
    contract_type_colors = {'Fixed': 'blue', 'Indexed': 'green', 'TakeOrPay': 'red'}

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each contract
    for idx, row in contract_data.iterrows():
        c = row['Contract']
        start = row['Start Month']
        duration = row['Duration']
        end = start + duration
        contract_type = row['Type']
        color = contract_type_colors.get(contract_type, 'black')
        ax.hlines(y=c, xmin=start, xmax=end, linewidth=5, color=color)

    # Setting labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Contract Number')
    ax.set_title('Contract Activation Over Time')
    ax.set_yticks(contract_data['Contract'])
    ax.set_yticklabels([f'Contract {int(c)}' for c in contract_data['Contract']])

    # Creating a custom legend
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=contract_type) for contract_type, color in contract_type_colors.items()]
    ax.legend(handles=legend_elements, title='Contract Types')

    plt.tight_layout()
    plt.show()
    # ... (Previous code remains the same up to data extraction)

    # Prepare data for Stacked Area Chart (Cumulative)
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    # Generate colors for contracts, using different colormaps for each contract type
    contract_type_colormaps = {
        'Fixed': cm.Blues,
        'TakeOrPay': cm.Reds,
        'Indexed': cm.Greens
    }

    # Order contracts by type and total amount (descending)
    contract_list = []
    for contract_type in ['Fixed', 'TakeOrPay', 'Indexed']:
        contracts_of_type = [c for c in contract_types if contract_types[c] == contract_type]
        # Sort contracts of the same type by total amount (descending)
        contracts_of_type.sort(key=lambda c: -contract_amounts[c])
        contract_list.extend(contracts_of_type)

    # Prepare colors for each contract
    contract_colors = {}
    for contract_type in contract_type_colormaps:
        cmap = contract_type_colormaps[contract_type]
        contracts_of_type = [c for c in contract_list if contract_types[c] == contract_type]
        num_contracts = len(contracts_of_type)
        for idx, c in enumerate(contracts_of_type):
            if num_contracts > 1:
                color = cmap(0.3 + 0.7 * idx / (num_contracts - 1))  # Varies from 0.3 to 1.0
            else:
                color = cmap(0.7)
            contract_colors[c] = color

    # Create a DataFrame with cumulative amounts
    # Convert monthly_data to DataFrame
    monthly_df = pd.DataFrame(monthly_data)

    # Pivot the data to have months as index and contracts as columns
    pivot_df = monthly_df.pivot(index='Month', columns='Contract', values='Amount').fillna(0)

    # Reorder columns according to the contract_list
    pivot_df = pivot_df[contract_list]

    # Prepare data for stackplot
    months = pivot_df.index.values
    amounts = pivot_df.values.T  # Transpose so each row corresponds to a contract

    # Prepare colors list in the same order as contracts
    colors = [contract_colors[c] for c in contract_list]

    # Plotting the Stacked Area Chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the stackplot with cumulative stacking
    ax.stackplot(months, amounts, labels=[f'Contract {int(c)}' for c in contract_list], colors=colors)

    # Plot the demand line
    ax.plot(months, nat_gas[:len(months)], label='Demand', color='black', linewidth=2)

    # Setting labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount Fulfilled')
    ax.set_title('Monthly Contract Fulfillment Over Time (Cumulative Stacked)')

    # Create custom legend for contract types
    legend_elements = []
    for contract_type, cmap in contract_type_colormaps.items():
        color = cmap(0.7)
        patch = Patch(facecolor=color, label=contract_type)
        legend_elements.append(patch)
    # Add demand line to legend
    legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Demand'))
    ax.legend(handles=legend_elements, title='Contract Types')

    plt.tight_layout()
    plt.show()

    import plotly.graph_objects as go
    import pandas as pd

    # Assuming 'monthly_data' is already prepared from previous steps

    # Prepare data for Plotly stacked area chart
    monthly_df = pd.DataFrame(monthly_data)

    # Pivot the data to have months as index and contracts as columns
    pivot_df = monthly_df.pivot(index='Month', columns='Contract', values='Amount').fillna(0)

    # Reorder columns according to the contract_list
    pivot_df = pivot_df[contract_list]

    # Prepare the contract types and colors for the plot
    contract_colors = []
    for c in contract_list:
        contract_type = contract_types.get(c, 'Unknown')
        base_color = contract_type_colormaps[contract_type]
        contract_colors.append(base_color(0.7))  # Adjust transparency as needed

    # Creating traces for each contract
    fig = go.Figure()

    for i, contract in enumerate(contract_list):
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[contract],
            mode='lines',
            name=f'Contract {contract}',
            stackgroup='one',  # Ensures stacking behavior
            line=dict(width=0.5),
            fillcolor=f'rgba({int(contract_colors[i][0]*255)}, {int(contract_colors[i][1]*255)}, {int(contract_colors[i][2]*255)}, 0.6)'
        ))

    # Adding the demand line
    fig.add_trace(go.Scatter(
        x=pivot_df.index,
        y=nat_gas[:len(pivot_df.index)],  # nat_gas demand for each month
        mode='lines',
        name='Demand',
        line=dict(color='black', width=2)
    ))

    # Update layout for better interactivity
    fig.update_layout(
        title="Monthly Contract Fulfillment Over Time (Cumulative Stacked)",
        xaxis_title="Month",
        yaxis_title="Amount Fulfilled",
        showlegend=True,
        hovermode="x unified",
        template="plotly_white"
    )

    # Show plot
    fig.show()
