#!/usr/bin/env python
import pandas as pd
import os

scen = "B"
# Path to the aggregated hourly CSV file.
base_dir = os.path.join(os.path.dirname(__file__), "results", scen)
aggregated_csv = os.path.join(base_dir, scen+"_hourly.csv")

# Read the aggregated hourly data
df = pd.read_csv(aggregated_csv)

# Group the data by 'Year'
grouped = df.groupby("Year")

# Prepare a list to hold summary information for each year
summary_list = []

for year, group in grouped:
    # For variables that are interval-based, we assume they are constant over the year,
    # so we take the first row's value.
    objective_cost = group["objective_expr"].iloc[0]  # Total Objective Cost in dollars
    fuel_cost = group["fuel_cost_NG"].iloc[0] + group["fuel_cost_H2"].iloc[0] + group["fuel_cost_BM"].iloc[0]
    carbon_cost = group["carbon_cost"].iloc[0]
    total_co2 = group["total_emissions_per_interval"].iloc[0] if "total_emissions_per_interval" in group.columns else float('nan')
    
    # Production throughput is assumed to be an hourly variable so we sum over all hours in the year.
    production_throughput = group["production_output"].sum()
    
    # Net revenue (if available) is assumed to be interval‐based, so we take the first row.
    if "total_costs" in group.columns and "total_revenues" in group.columns:
        net_revenue = group["total_revenues"].iloc[0] - group["total_costs"].iloc[0]
    else:
        net_revenue = float('nan')
    
    # Additional revenue metrics (modeled as intervals, so constant over the year)
    if "elec_sold_expr" in group.columns:
        elec_revenue = group["elec_sold_expr"].iloc[0]
    else:
        elec_revenue = float('nan')
    
    if "heat_sold_expr" in group.columns:
        heat_revenue = group["heat_sold_expr"].iloc[0]
    else:
        heat_revenue = float('nan')
    
    if "ancillary_revenue" in group.columns:
        ancillary_revenue = group["ancillary_revenue"].iloc[0]
    else:
        ancillary_revenue = float('nan')
    
    # Convert dollar values to millions ($M) where applicable.
    summary_list.append({
        "Year": year,
        "Total Objective Cost ($M)": objective_cost / 1e6,
        "Fuel Cost ($M)": fuel_cost / 1e6,
        "Carbon Compliance Cost ($M)": carbon_cost / 1e6,
        "Total CO2 Emissions (tonnes)": total_co2,
        "Production Throughput (tons/year)": production_throughput,
        "Net Revenue ($M)": net_revenue / 1e6 if not pd.isna(net_revenue) else net_revenue,
        "Total Electricity Revenue ($M)": elec_revenue / 1e6 if not pd.isna(elec_revenue) else elec_revenue,
        "Total Heat Revenue ($M)": heat_revenue / 1e6 if not pd.isna(heat_revenue) else heat_revenue,
        "Total Ancillary Revenue ($M)": ancillary_revenue / 1e6 if not pd.isna(ancillary_revenue) else ancillary_revenue
    })

# Create a DataFrame for the summary and sort by Year.
summary_df = pd.DataFrame(summary_list).sort_values("Year")

# Save the summary to a CSV file (this CSV can later be used to produce your LaTeX table)
output_csv = os.path.join(base_dir, "master_kpi_summary.csv")
summary_df.to_csv(output_csv, index=False)

print(f"Saved KPI summary to {output_csv}")
