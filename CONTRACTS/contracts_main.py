from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys
from pyomo.environ import value

# Append the configuration path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options (ensure 'solver_options.py' is accessible)
from solver_options import get_solver

current_dir = os.path.dirname(__file__)

# Construct paths to the data files
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands_monthly_30.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.csv'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.csv'))

# Read the Excel files
# Read the Excel files
demands = pd.read_excel(demands_path, nrows=361)
markets_monthly = pd.read_csv(markets_monthly_path, nrows=361)

electricity_demand = demands["elec"].to_numpy()
nat_gas = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
electricity_market = markets_monthly["Electricity Price ($/kWh)"].to_numpy() 
electricity_market_sold = markets_monthly["Electricity Price ($/kWh)"].to_numpy() * 0.8
carbon_market = markets_monthly["Carbon Credit Price ($/tonne CO2)"].to_numpy() 
NG_market = markets_monthly["Natural Gas Price ($/kWh)"].to_numpy() 
NG_market_monthly = markets_monthly["Natural Gas Price ($/kWh)"].to_numpy()
heat_market_sold = markets_monthly["Natural Gas Price ($/kWh)"].to_numpy() * 0.8
H2_market = markets_monthly["Hydrogen Price ($/kWh)"].to_numpy() 
BM_market = markets_monthly["Biomass Price ($/kWh)"].to_numpy()

total_times = 60  # Adjust based on your data
no_contract = 5   # Adjust based on your preference
time_limit = 30
bound_duration = 60  # Maximum duration is the total time horizon

risk_appetite_threshold = 500 * no_contract  # Adjusted to avoid infeasibility

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # Sets
    total_time = total_times
    model.MONTHS = RangeSet(0, total_time - 1)
    model.CONTRACTS = RangeSet(0, no_contract - 1)
    model.ContractTypes = Set(initialize=['Fixed', 'Indexed', 'TakeOrPay'])

    # Parameters
    min_price = min(NG_market_monthly) * 0.5
    max_price = max(NG_market_monthly) * 2

    # Big-M value for constraints
    M = 1E7

    # Upper bound U for ContractAmount
    U = 1E7  # Adjusted based on maximum expected amount
    min_duration = 6
    # Variables
    model.ContractStart = Var(model.CONTRACTS, model.MONTHS, within=Binary)
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(min_duration, bound_duration))
    model.ContractActive = Var(model.CONTRACTS, model.MONTHS, within=Binary)
    model.ContractType = Var(model.CONTRACTS, model.ContractTypes, within=Binary)
    model.ContractAmount = Var(model.CONTRACTS, model.MONTHS, within=NonNegativeReals, bounds=(0, U))
    model.ContractPrice = Var(model.CONTRACTS, model.MONTHS, within=NonNegativeReals)
    model.ContractStartPrice = Var(model.CONTRACTS, within=NonNegativeReals)
    model.ContractAmountActive = Var(model.CONTRACTS, model.MONTHS, within=NonNegativeReals, bounds=(0, U))
    model.TotalCost = Var(model.CONTRACTS, model.MONTHS, within=NonNegativeReals)
    model.ContractEnd = Var(model.CONTRACTS, model.MONTHS, within=Binary)
    model.risk_score = Var(within=NonNegativeReals)

    # Constraints

    # Ensure only one contract type is selected per contract
    def contract_type_rule(model, c):
        return sum(model.ContractType[c, t] for t in model.ContractTypes) == 1
    model.ContractTypeConstraint = Constraint(model.CONTRACTS, rule=contract_type_rule)

    # Each contract can start at most once
    def contract_single_start_rule(model, c):
        return sum(model.ContractStart[c, m] for m in model.MONTHS) <= 1
    model.ContractSingleStartCon = Constraint(model.CONTRACTS, rule=contract_single_start_rule)

    # Set Contract Start Price
    def set_contract_start_price_rule(model, c):
        return model.ContractStartPrice[c] == sum(
            NG_market_monthly[m] * model.ContractStart[c, m] for m in model.MONTHS
        )
    model.SetContractStartPrice = Constraint(model.CONTRACTS, rule=set_contract_start_price_rule)

    # Contract Price Setting
    def contract_price_setting_rule(model, c, m):
        fixed_price = model.ContractType[c, 'Fixed'] * model.ContractStartPrice[c] * 1.1  # 10% premium
        # Indexed contracts at market price (no premium)
        indexed_price = model.ContractType[c, 'Indexed'] * NG_market_monthly[m]
        # TakeOrPay contracts at a larger discount (e.g., 15% discount)
        take_or_pay_price = model.ContractType[c, 'TakeOrPay'] * model.ContractStartPrice[c] * 0.9  # 20% discount
        return model.ContractPrice[c, m] == fixed_price + indexed_price + take_or_pay_price

    model.ContractPriceSetting = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_price_setting_rule)

    # Contract Duration Definition
    def contract_duration_rule(model, c):
        return model.ContractDuration[c] == sum(model.ContractActive[c, m] for m in model.MONTHS)
    model.ContractDurationCon = Constraint(model.CONTRACTS, rule=contract_duration_rule)

    # Activation Constraints
    # Ensure that once a contract starts, it remains active for its duration
    def activation_rule(model, c, m):
        if m == 0:
            return model.ContractActive[c, m] == model.ContractStart[c, m]
        else:
            return model.ContractActive[c, m] >= model.ContractActive[c, m - 1] - model.ContractEnd[c, m - 1] + model.ContractStart[c, m]
    model.ContractActivationCon = Constraint(model.CONTRACTS, model.MONTHS, rule=activation_rule)

    # Contract End Constraints
    def contract_end_rule(model, c, m):
        if m == total_time - 1:
            # For the last month
            return model.ContractEnd[c, m] >= model.ContractActive[c, m] - 0
        else:
            return model.ContractEnd[c, m] >= model.ContractActive[c, m] - model.ContractActive[c, m + 1]
    model.ContractEndCon = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_end_rule)

    # Ensure that the contract ends after the duration
    def contract_end_duration_rule(model, c):
        return sum(model.ContractEnd[c, m] for m in model.MONTHS) <= 1
    model.ContractEndDurationCon = Constraint(model.CONTRACTS, rule=contract_end_duration_rule)

    # Ensure that the contract remains active for the specified duration
    def contract_duration_active_rule(model, c):
        return sum(model.ContractActive[c, m] for m in model.MONTHS) == model.ContractDuration[c]
    model.ContractDurationActiveCon = Constraint(model.CONTRACTS, rule=contract_duration_active_rule)

    # Linking ContractActive with ContractStart
    def contract_active_link_rule(model, c, m):
        return model.ContractActive[c, m] <= sum(model.ContractStart[c, s] for s in model.MONTHS if s <= m)
    model.ContractActiveLinkCon = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_active_link_rule)

    # Correct Linearization of ContractAmountActive = ContractAmount * ContractActive
    def contract_amount_active_upper(model, c, m):
        return model.ContractAmountActive[c, m] <= model.ContractAmount[c, m]
    model.ContractAmountActiveUpper = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_amount_active_upper)

    def contract_amount_active_binary(model, c, m):
        return model.ContractAmountActive[c, m] <= model.ContractActive[c, m] * U
    model.ContractAmountActiveBinary = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_amount_active_binary)

    def contract_amount_active_lower(model, c, m):
        return model.ContractAmountActive[c, m] >= model.ContractAmount[c, m] - (1 - model.ContractActive[c, m]) * U
    model.ContractAmountActiveLower = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_amount_active_lower)

    def contract_amount_active_nonneg(model, c, m):
        return model.ContractAmountActive[c, m] >= 0
    model.ContractAmountActiveNonneg = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_amount_active_nonneg)

    # Ensure ContractAmount is zero when contract is inactive
    def contract_amount_zero_when_inactive(model, c, m):
        return model.ContractAmount[c, m] <= model.ContractActive[c, m] * U
    model.ContractAmountZeroInactiveCon = Constraint(model.CONTRACTS, model.MONTHS, rule=contract_amount_zero_when_inactive)

    # Demand Fulfillment Constraint
    def demand_fulfillment_rule(model, m):
        return sum(model.ContractAmountActive[c, m] for c in model.CONTRACTS) >= nat_gas[m]
    model.DemandFulfillmentCon = Constraint(model.MONTHS, rule=demand_fulfillment_rule)

    # Adjusted Minimum Contract Duration

    def min_contract_duration_rule(model, c):
        return model.ContractDuration[c] >= min_duration * sum(model.ContractStart[c, m] for m in model.MONTHS)
    model.MinContractDurationCon = Constraint(model.CONTRACTS, rule=min_contract_duration_rule)

    # For 'TakeOrPay' Contracts (minimum amount)
    def min_fulfillment_takeorpay_rule(model, c, m):
        return model.ContractAmount[c, m] >= 670 * model.ContractActive[c, m] - (1 - model.ContractType[c, 'TakeOrPay']) * M
    model.MinFulfillmentTakeOrPayCon = Constraint(model.CONTRACTS, model.MONTHS, rule=min_fulfillment_takeorpay_rule)

    # Adjusted Maximum Fulfillment Constraints per Contract Type
    def max_fulfillment_fixed_rule(model, c, m):
        return model.ContractAmount[c, m] <= U * model.ContractActive[c, m] + (1 - model.ContractType[c, 'Fixed']) * M
    model.MaxFulfillmentFixedCon = Constraint(model.CONTRACTS, model.MONTHS, rule=max_fulfillment_fixed_rule)

    def max_fulfillment_indexed_rule(model, c, m):
        return model.ContractAmount[c, m] <= U * model.ContractActive[c, m] + (1 - model.ContractType[c, 'Indexed']) * M
    model.MaxFulfillmentIndexedCon = Constraint(model.CONTRACTS, model.MONTHS, rule=max_fulfillment_indexed_rule)

    def max_fulfillment_takeorpay_rule(model, c, m):
        return model.ContractAmount[c, m] <= 670 * model.ContractActive[c, m] + (1 - model.ContractType[c, 'TakeOrPay']) * M
    model.MaxFulfillmentTakeOrPayCon = Constraint(model.CONTRACTS, model.MONTHS, rule=max_fulfillment_takeorpay_rule)

    # **Modified Constraint**: Limit the number of Indexed contracts active at the same time
    def limit_indexed_contracts_rule(model, m):
        return sum(model.ContractActive[c, m] * model.ContractType[c, 'Indexed'] for c in model.CONTRACTS) <= 1
    model.LimitIndexedContracts = Constraint(model.MONTHS, rule=limit_indexed_contracts_rule)

    # Risk Score Calculation
    def risk_score_rule(model):
        return model.risk_score == sum(
            (1 * model.ContractType[c, 'Fixed'] +
             3 * model.ContractType[c, 'Indexed'] +
             2 * model.ContractType[c, 'TakeOrPay']) *
            model.ContractDuration[c]
            for c in model.CONTRACTS
        )
    model.RiskScoreCon = Constraint(rule=risk_score_rule)
    # Risk Appetite Threshold
    def risk_appetite_rule(model):
        return model.risk_score <= risk_appetite_threshold
    model.RiskAppetiteCon = Constraint(rule=risk_appetite_rule)

    # Contract Type Binary Constraints
    def contract_type_binary_rule(model, c, t):
        return model.ContractType[c, t] <= 1
    model.ContractTypeBinaryCon = Constraint(model.CONTRACTS, model.ContractTypes, rule=contract_type_binary_rule)

    # Total Cost Calculation
    def total_cost_constraint_rule(model, c, m):
        return model.TotalCost[c, m] == model.ContractAmountActive[c, m] * model.ContractPrice[c, m]
    model.TotalCostConstraint = Constraint(model.CONTRACTS, model.MONTHS, rule=total_cost_constraint_rule)

    # Objective function
    def objective_rule(model):
        return sum(model.TotalCost[c, m] for c in model.CONTRACTS for m in model.MONTHS)
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Solver configuration
  # Solver configuration
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.options['MIPGap'] = 0.01          # Tighter optimality gap
    solver.options['Threads'] = 32            # Use all available threads
    solver.options['MIPFocus'] = 1           # Focus on finding feasible solutions quickly
    solver.options['Presolve'] = 2           # Aggressive presolve
    solver.options['Heuristics'] = 1         # Increase heuristic search
    solver.options['Cuts'] = 2               # Aggressive cut generation
    solver.options['Symmetry'] = 2           # Aggressive symmetry detection
    solver.options['Method'] = 2             # Use Barrier method for LP relaxations
    solver.options["NonConvex"] = 2

    # Solve the model
    results = solver.solve(model, tee=True, symbolic_solver_labels=False)

    # Check solver status
    if results.solver.termination_condition != TerminationCondition.optimal and \
       results.solver.termination_condition != TerminationCondition.feasible:
        print("Solver did not find an optimal solution.")
    else:
        print("Solver found an optimal solution.")

    return model




def verify_contract_data(contract_data, model, total_months):
    """
    Verifies if the contract data has gaps or incorrect activations based on the Pyomo model.

    Parameters:
        contract_data (DataFrame): A DataFrame with contract information (start, duration, type, etc.).
        model (Pyomo Model): The Pyomo model containing contract activation variables.
        total_months (int): The total number of months in the model.

    Returns:
        bool: True if no issues found, False if issues are found.
    """
    issues_found = False

    print("Verifying contract data...")

    for idx, row in contract_data.iterrows():
        c = row['Contract']
        start = row['Start Month']
        duration = row['Duration']
        contract_type = row['Type']

        # Print contract summary
        print(f"\nContract {c} (Type: {contract_type})")
        print(f"Expected Start Month: {start}, Duration: {duration}")
        
        # Ensure contract starts at the correct month
        start_found = False
        for m in range(total_months):
            if value(model.ContractStart[c, m]) >= 0.99:
                if m != start:
                    print(f"  -> Warning: Model start month {m} does not match contract data start month {start}")
                    issues_found = True
                start_found = True
                break

        if not start_found:
            print(f"  -> Error: No start found for contract {c}!")
            issues_found = True

        # Check contract duration and for gaps in contract activity
        active_months = 0
        for m in range(start, start + int(duration)):
            if value(model.ContractActive[c, m]) >= 0.99:
                active_months += 1
            else:
                print(f"  -> Error: Gap in contract {c} activity at month {m}!")
                issues_found = True

        if active_months != duration:
            print(f"  -> Error: Active months ({active_months}) do not match contract duration ({duration})")
            issues_found = True

    if issues_found:
        print("\nVerification complete. Issues were found in the contract data.")
    else:
        print("\nVerification complete. No issues found in the contract data.")

    return not issues_found

if __name__ == '__main__':
    model = pyomomodel()
    # Data Extraction
    start_months = {}
    durations = {}
    contract_types = {}
    contract_amounts = {}
    contract_prices = {}
    monthly_data = []

    # Loop over all contracts
    for c in model.CONTRACTS:
        # Find the start month
        start_month = None
        for m in model.MONTHS:
            if value(model.ContractStart[c, m]) >= 0.99:
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
        total_amount = sum(value(model.ContractAmount[c, m]) for m in model.MONTHS if value(model.ContractActive[c, m]) >= 0.99)
        contract_amounts[c] = total_amount

        # Get contract start price
        start_price = value(model.ContractStartPrice[c])
        contract_prices[c] = start_price

        # Collect monthly data for this contract
        for m in model.MONTHS:
            if value(model.ContractActive[c, m]) >= 0.99:
                amount = value(model.ContractAmount[c, m])
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

    # Optionally, verify the contract data
    total_months = total_times
    is_valid = verify_contract_data(contract_data, model, total_months)

    # Optionally, save the data to a CSV file
    contract_data.to_csv('contract_data.csv', index=False)

    # Plotting the Contract Activation Over Time
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    # Interactive Plot using Plotly
    import plotly.graph_objects as go

    # Prepare the contract types and colors for the plot
    contract_colors_rgba = []
    for c in contract_list:
        contract_type = contract_types.get(c, 'Unknown')
        base_color = contract_type_colormaps[contract_type]
        rgba_color = base_color(0.7)
        rgba_color_str = f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})'
        contract_colors_rgba.append(rgba_color_str)

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
            fillcolor=contract_colors_rgba[i]
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
