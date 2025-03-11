#!/usr/bin/env python
import pandas as pd
import os

# Path to the aggregated hourly CSV file.
# Adjust this path if needed; here we assume it is saved in the results folder.
base_dir = os.path.join(os.path.dirname(__file__), "results", "base_case")
aggregated_csv = os.path.join(base_dir, "base_case_hourly.csv")

# Read the aggregated hourly data
df = pd.read_csv(aggregated_csv)

# Group the data by 'Year'
grouped = df.groupby("Year")

# Prepare a list to hold summary information for each year
summary_list = []

for year, group in grouped:
    # For variables saved from interval-based calculations, we assume they are constant over the year.
    # So, we take the first row.
    objective_cost = group["objective_expr"].iloc[0]  # Total Objective Cost in dollars
    fuel_cost = (group["fuel_cost_NG"].iloc[0] +
                 group["fuel_cost_H2"].iloc[0] +
                 group["fuel_cost_BM"].iloc[0])  # Total Fuel Cost in dollars
    carbon_cost = group["carbon_cost"].iloc[0]  # Carbon Compliance Cost in dollars

    # For CO₂ emissions, if you saved an interval value (e.g. "total_emissions_per_interval"),
    # we take the first value; otherwise, set to NaN.
    total_co2 = group["total_emissions_per_interval"].iloc[0] if "total_emissions_per_interval" in group.columns else float('nan')

    # Production throughput is assumed to be an hourly variable (e.g. "production_output")
    # so we sum over all hours in the year.
    production_throughput = group["production_output"].sum()

    # If you saved total cost and revenue columns, compute net revenue as:
    if "total_costs" in group.columns and "total_revenues" in group.columns:
        net_revenue = group["total_revenues"].iloc[0] - group["total_costs"].iloc[0]
    else:
        net_revenue = float('nan')

    # Convert dollar values to millions ($M) where applicable.
    summary_list.append({
        "Year": year,
        "Total Objective Cost ($M)": objective_cost / 1e6,
        "Fuel Cost ($M)": fuel_cost / 1e6,
        "Carbon Compliance Cost ($M)": carbon_cost / 1e6,
        "Total CO2 Emissions (tonnes)": total_co2,
        "Production Throughput (tons/year)": production_throughput,  # adjust conversion if needed
        "Net Revenue ($M)": net_revenue / 1e6 if not pd.isna(net_revenue) else net_revenue
    })

# Create a DataFrame for the summary
summary_df = pd.DataFrame(summary_list)

# Optionally, sort the summary by year
summary_df = summary_df.sort_values("Year")

# Save the summary to a CSV file (this CSV can later be used to produce your LaTeX table)
output_csv = os.path.join(base_dir, "master_kpi_summary.csv")
summary_df.to_csv(output_csv, index=False)

print(f"Saved KPI summary to {output_csv}")
