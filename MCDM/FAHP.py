# Systems
import numpy as np
import plotly.graph_objects as go

def fill_lower_triangle(matrix):
    """
    Fills in the lower triangle of a matrix with the reciprocal 
    of the upper triangle values. 
    The diagonal remains unchanged.
    """
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i+1, cols):
            matrix[j][i] = 1/matrix[i][j]
    return matrix

class ChangFuzzyAHP:
    def __init__(self):
        None
    def chang_extent_analysis(self, matrix, weight):
        """Perform Chang's extent analysis on the given matrix."""
        # Initialize
        n = matrix.shape[0]
        S = np.zeros((n, 3))
        extent_matrix = np.zeros((n, n, 3))
        
        # Calculate the extent matrix for each object vs. all other objects
        for i in range(n):
            for j in range(n):
                extent_matrix[i, j] = matrix[i, j] * weight
        
        # Calculate the synthetic extent value for each object
        for i in range(n):
            S[i] = np.sum(extent_matrix[i], axis=0) / n
        
        # Calculate the degree of possibility
        V = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if S[i][0] >= S[j][2]:
                    V[i, j] = 1
                elif S[i][2] < S[j][0]:
                    V[i, j] = 0
                else:
                    V[i, j] = (S[i][2] - S[j][0]) / (S[j][1] - S[j][0] + S[i][1] - S[i][0])
        
        # Calculate the final weights
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = 1 / np.sum(V[:, i])
        
        # Normalize the weights
        normalized_weights = weights / np.sum(weights)
        
        return normalized_weights

    def rank_criteria(self, matrix):
        """Rank the criteria based on their importance using Chang's extent analysis."""
        return self.chang_extent_analysis(matrix, np.array([1, 1, 1]))
    
    def rank_systems(self, criteria_weights, matrices):
        """Rank systems based on Chang's method."""
        total_weights = np.zeros(len(systems))
        for i in range(len(matrices)):
            weights = self.chang_extent_analysis(matrices[i], criteria_weights[i])
            total_weights += weights
        ranked_systems = [x for _, x in sorted(zip(total_weights, systems), reverse=True)]
        return ranked_systems

# The ChangFuzzyAHP class has methods for Chang's extent analysis and ranking criteria and systems.

systems = ["Electrifiction", "Natural Gas", "Hydrogen", "Biomass", "CCS"]

# Criteria and Sub-criteria
criteria = ["Emissions", "Economics", "Safety", "Social"]
sub_criteria = {
    "Emissions": ["CO2", "Other", "Indirect"],
    "Economics": ["CAPEX", "OPEX", "Maintenance"],
    "Safety": ["Operational", "Environmental", "Resilience"],
    "Social": ["PR", "JOB", "Expertise"]
}


# Linguistic variables as Triangular Fuzzy Numbers (TFNs)
equal = np.array([1, 1, 1])
n1 = equal
n2 = np.array([1, 2, 3], dtype=object)
n3 = np.array([2, 3, 4], dtype=object)
n4 = np.array([3, 4, 5], dtype=object)
n5 = np.array([4, 5, 6], dtype=object)
n6 = np.array([5, 6, 7], dtype=object)
n7 = np.array([6, 7, 8], dtype=object)

# Placeholder Fuzzy Pairwise Comparison Matrix for Criteria 
criteria_matrix = np.array([
    [equal, n2, 1/n3, 1/n2],
    [1/n2, equal, 1/n4, 1/n3],
    [n3, n4, equal, n2],
    [n2, n3, 1/n2, equal]
])


# Placeholder matrices for sub-criteria under each main criterion 

# Safety Sub-Criteria Matrix
sub_criteria_matrix_safety = np.array([
    [equal, n2, 1/n2],
    [1/n2, equal, 1/n3],
    [n2, n3, equal]
])

# Economics Sub-Criteria Matrix
sub_criteria_matrix_economics = np.array([
    [equal, 1/n2, n2],
    [1/equal, equal, n3],
    [1/equal, 1/equal, equal]
])

# Emissions Sub-Criteria Matrix
sub_criteria_matrix_emissions = np.array([
    [equal, n3, n2],
    [1/n3, equal, 1/n2],
    [1/n2, n2, equal]
])

# Social Sub-Criteria Matrix
sub_criteria_matrix_social = np.array([
    [equal, 1/n2, n2],
    [n2, equal, n3],
    [1/n2, 1/n3, equal]
])

sub_criteria_matrices = {
    "Emissions": sub_criteria_matrix_emissions,
    "Economics": sub_criteria_matrix_economics,
    "Safety": sub_criteria_matrix_safety,
    "Social": sub_criteria_matrix_social
}

sub_criteria_matrices

chang_fahp = ChangFuzzyAHP()

# Computing the weights for main criteria using Chang's method
criteria_weights_chang = chang_fahp.rank_criteria(criteria_matrix)

# Computing the weights for sub-criteria under each main criterion using Chang's method
sub_criteria_weights_chang = {}

for crit in criteria:
    sub_criteria_weights_chang[crit] = chang_fahp.rank_criteria(sub_criteria_matrices[crit])

sub_criteria_weights_chang

# Placeholder matrices for systems under each sub-criterion (introducing some variety for demonstration)

test = fill_lower_triangle(np.array([
    [n1, n2, 2, 4, 8],
    [0, n1, 2, 5, 8],
    [0, 0, n1, 2, 4],
    [0, 0, 0, n1, 4],
    [0, 0, 0, 0, n1]
]))
print(test)

co2 = fill_lower_triangle(np.array([
    [n1, n2, 1/n2, 1/n3, n3],
    [0, n1, 1/n4, 1/n5, 1/n2],
    [0, 0, n1, 1/n2, n3],
    [0, 0, 0, n1, n4],
    [0, 0, 0, 0, n1]
]))

other = fill_lower_triangle(np.array([
    [n1, n3, 1/n2, 1/n3, n2],
    [0, n1, 1/n4, 1/n5, 1/n2],
    [0, 0, n1, 1/n2, n3],
    [0, 0, 0, n1, n4],
    [0, 0, 0, 0, n1]
]))

indirect = fill_lower_triangle(np.array([
    [n1, n3, n2, 1/n2, n3],
    [0, n1, 1/n2, 1/n4, n2],
    [0, 0, n1, 1/n3, n2],
    [0, 0, 0, n1, n4],
    [0, 0, 0, 0, n1]
]))

capex = fill_lower_triangle(np.array([
    [n1, 1/n2, 1/n3, n3, 1/n2],
    [0, n1, 1/n4, n2, n2],
    [0, 0, n1, n5, n3],
    [0, 0, 0, n1, n4],
    [0, 0, 0, 0, n1]
]))

opex = fill_lower_triangle(np.array([
    [n1, n2, n2, 1/n2, n3],
    [0, n1, n1, 1/n3, 1/n2],
    [0, 0, n1, 1/n3, 1/n2],
    [0, 0, 0, n1, n2],
    [0, 0, 0, 0, n1]
]))

maintenance = fill_lower_triangle(np.array([
    [n1, n2, 1/n2, 1/n3, n3],
    [0, n1, 1/n3, 1/n4, 1/n2],
    [0, 0, n1, 1/n5, n2],
    [0, 0, 0, n1, n3],
    [0, 0, 0, 0, n1]
]))

operational = fill_lower_triangle(np.array([
    [n1, n2, 1/n2, n3, 1/n3],
    [0, n1, 1/n4, n2, 1/n5],
    [0, 0, n1, n5, 1/n6],
    [0, 0, 0, n1, 1/n7],
    [0, 0, 0, 0, n1]
]))

environmental = fill_lower_triangle(np.array([
    [n1, n2, 1/n2, 1/n3, 1/n4],
    [0, n1, 1/n3, 1/n4, 1/n5],
    [0, 0, n1, 1/n5, 1/n6],
    [0, 0, 0, n1, n2],
    [0, 0, 0, 0, n1]
]))

resilience = fill_lower_triangle(np.array([
    [n1, n2, 1/n2, 1/n3, 1/n4],
    [0, n1, 1/n3, 1/n4, 1/n5],
    [0, 0, n1, 1/n5, 1/n6],
    [0, 0, 0, n1, 1/n7],
    [0, 0, 0, 0, n1]
]))

PR = fill_lower_triangle(np.array([
    [n1, 1/n2, n2, 1/n3, n3],
    [0, n1, n3, 1/n4, n4],
    [0, 0, n1, 1/n5, 1/n6],
    [0, 0, 0, n1, 1/n7],
    [0, 0, 0, 0, n1]
]))

job = fill_lower_triangle(np.array([
    [n1, 1/n2, n2, 1/n3, n3],
    [0, n1, n3, 1/n4, n4],
    [0, 0, n1, 1/n5, n5],
    [0, 0, 0, n1, n6],
    [0, 0, 0, 0, n1]
]))

expertise = fill_lower_triangle(np.array([
    [n1, 1/n2, 1/n3, 1/n4, 1/n5],
    [0, n1, 1/n3, 1/n4, 1/n5],
    [0, 0, n1, 1/n5, 1/n6],
    [0, 0, 0, n1, 1/n7],
    [0, 0, 0, 0, n1]
]))

# Using system_matrix_safety_1 as a placeholder for the rest for simplicity
system_matrices = {
    "Emissions": {
        "CO2": co2,
        "Other": other,
        "Indirect": indirect
    },
    "Economics": {
        "CAPEX": capex,
        "OPEX": opex,
        "Maintenance": maintenance
    },
    "Safety": {
        "Operational": operational,
        "Environmental": environmental,
        "Resilience": resilience
    },
    "Social": {
        "PR": PR,
        "JOB": job,
        "Expertise": expertise
    }
}

# Computing the rankings for the systems based on Chang's method

# Placeholder to accumulate the weights for each system
total_weights = np.zeros(len(systems))

# Iterate over main criteria
for crit in criteria:
    # Get the weight of the main criterion
    main_weight = criteria_weights_chang[criteria.index(crit)]
    
    # Iterate over sub-criteria under the main criterion
    for sub_crit in sub_criteria[crit]:
        # Get the weight of the sub-criterion
        sub_weight = sub_criteria_weights_chang[crit][sub_criteria[crit].index(sub_crit)]
        
        # Calculate the system weights for this sub-criterion using Chang's method
        system_weights = chang_fahp.chang_extent_analysis(system_matrices[crit][sub_crit], [1, 1, 1])
        
        # Accumulate the total weights considering the weights of main and sub-criteria
        total_weights += main_weight * sub_weight * system_weights

# Normalize the total weights
normalized_weights = total_weights / np.sum(total_weights)

# Rank the systems based on the normalized weights
ranked_systems_chang = [x for _, x in sorted(zip(normalized_weights, systems), reverse=True)]

print(systems)
print(normalized_weights)
print("----------------------")
print(ranked_systems_chang)
print(np.sort(normalized_weights))
print("----------------------")
print(criteria)
print(criteria_weights_chang)
print(sub_criteria_weights_chang)

def calculate_synthetic_extent(matrix):
    """Calculate the fuzzy synthetic extent values for a fuzzy pairwise comparison matrix."""
    n = matrix.shape[0]
    S = np.zeros((n, 3))
    extent_matrix = np.zeros((n, n, 3))
    
    # Calculate the extent matrix for each object vs. all other objects
    for i in range(n):
        for j in range(n):
            extent_matrix[i, j] = matrix[i, j] 
    
    # Calculate the synthetic extent value for each object
    for i in range(n):
        S[i] = np.sum(extent_matrix[i], axis=0) / n
        
    return S

def normalize_synthetic_extent(S):
    """Normalize the fuzzy synthetic extent values to get the fuzzy weight vector."""
    total = np.sum(S, axis=0)
    return S / total

# Calculate fuzzy synthetic extent values and normalize them to get the fuzzy weight vector
S_criteria = calculate_synthetic_extent(criteria_matrix)
fuzzy_weights_criteria = normalize_synthetic_extent(S_criteria)

fuzzy_weights_criteria

def fuzzy_matrix_multiplication(matrix, vector):
    """Multiply a fuzzy matrix by a fuzzy vector."""
    n = matrix.shape[0]
    result = np.zeros((n, 3))
    
    for i in range(n):
        for j in range(3):
            result[i, j] = np.sum(matrix[i, :, j] * vector[:, 1])
            
    return result

# Compute the fuzzy consistency vector
fuzzy_consistency_vector = fuzzy_matrix_multiplication(criteria_matrix, fuzzy_weights_criteria)

fuzzy_consistency_vector


def calculate_gci(fuzzy_weights, fuzzy_consistency_vector):
    """Calculate the Geometric Consistency Index (GCI) for fuzzy weights and consistency vector."""
    n = fuzzy_weights.shape[0]
    
    # Compute the distance between the fuzzy consistency vector and the fuzzy weight vector
    distance = np.sum((fuzzy_consistency_vector - fuzzy_weights) ** 2, axis=1)
    
    # Normalize this distance
    normalized_distance = distance / n
    
    # The GCI is the square root of the sum of the normalized distances
    gci = np.sqrt(np.sum(normalized_distance))
    
    return gci

# Compute the GCI
gci = calculate_gci(fuzzy_weights_criteria, fuzzy_consistency_vector)

print(gci)


import plotly.graph_objects as go
import numpy as np

# Colors for the bars as provided
colors_plotly = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Data and sub-criteria labels
data = sub_criteria_weights_chang

sub_criteria_labels = {
    'Emissions': ['CO2', 'Other', 'Indirect'],
    'Economics': ['CAPEX', 'OPEX', 'Maintenance'],
    'Safety': ['Operational', 'Environmental', 'Resilience'],
    'Social': ['Public Relations', 'Job Creation', 'Expertise']
}

# Sort the weights and sub-criteria labels for each main criterion
sorted_weights = {}
sorted_sub_criteria = {}
for label in data.keys():
    sorted_indices = np.argsort(data[label])[::-1]
    sorted_weights[label] = data[label][sorted_indices]
    sorted_sub_criteria[label] = np.array(sub_criteria_labels[label])[sorted_indices]

# Colors for the bars
colors_plotly = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# Text font properties for bars
text_font_properties = dict(size=16, color='white', family="Arial, bold")

# Initialize the figure



# Main criteria weights
# Main criteria weights
gap_value = 0.05

# Donut Chart with enhanced aesthetics
fig_donut = go.Figure(data=[go.Pie(
    labels=criteria, 
    values=criteria_weights_chang, 
    hole=0.3, 
    textinfo='label+percent', 
    pull=[0.1, 0.1, 0.1, 0.1],
    insidetextorientation='radial',
    textfont=dict(size=30, family="Georgia, bold", color="black")
)])

fig_donut.update_layout(
    title_text="Main Criteria Weights",
    title_font=dict(size=40, family="Georgia, bold", color="black"),
    legend_title_text="Criteria",
    legend_title_font=dict(size=30, family="Georgia, bold", color="black"),
    legend_font=dict(size=30, family="Georgia"),
    template='seaborn'
)

# Display the figure
fig_donut.show()


import numpy as np
import plotly.graph_objects as go

sub_criteria_labels = {
    'Emissions': ['CO2', 'Other', 'Indirect'],
    'Economics': ['CAPEX', 'OPEX', 'Maintenance'],
    'Safety': ['Operational', 'Environmental', 'Resilience'],
    'Social': ['Public Relations', 'Job Creation', 'Expertise']
}

# Data for the bar chart
bar_data = sub_criteria_weights_chang

# Overall weightings for the main criteria
overall_weights = criteria_weights_chang

# Adjust the bar data values based on the overall weightings
sub_data = {}
for i, (criterion, values) in enumerate(bar_data.items()):
    sub_data[criterion] = values * overall_weights[i]

# Flatten data to create a single list of values and labels
flat_labels = []
flat_values = []
flat_colors = []

# Define a color map for the main criteria
color_map = {
    'Emissions': colors_plotly[2],
    'Economics': colors_plotly[0],
    'Safety': colors_plotly[1],
    'Social': colors_plotly[3]  # Reusing color for demonstration, this can be changed
}


# Properly group the sub-criteria based on the specified order
correct_order_labels = []
correct_order_values = []
correct_order_colors = []
correct_order_pulls = []

# Specify the correct order for main criteria and sub-criteria
ordered_criteria = {
    'Economics': ['CAPEX', 'OPEX', 'Maintenance'],
    'Emissions': ['CO2', 'Other', 'Indirect'],
    'Safety': ['Operational', 'Environmental', 'Resilience'],
    'Social': ['Public Relations', 'Job Creation', 'Expertise']
}

# Populate the lists based on the correct order
for main_criterion, sub_criterion_order in ordered_criteria.items():
    for sub_criterion in sub_criterion_order:
        idx = sub_criteria_labels[main_criterion].index(sub_criterion)
        correct_order_labels.append(sub_criterion)
        correct_order_values.append(sub_data[main_criterion][idx])
        correct_order_colors.append(color_map[main_criterion])
        correct_order_pulls.append(0.1)

# Generate the donut chart with the proper grouping
fig_correct_order_donut = go.Figure(data=[go.Pie(
    labels=correct_order_labels,
    values=correct_order_values,
    hole=0.3,
    textinfo='label+percent',
    pull=correct_order_pulls,
    marker=dict(colors=correct_order_colors),
    insidetextorientation='radial',
    textfont=dict(size=25, family="Georgia, bold", color="black"),
    sort=False
)])

fig_correct_order_donut.update_layout(
    title_text="Sub-Criteria Weights Ordered",
    title_font=dict(size=24, family="Georgia, bold", color="black"),
    legend_title_text="Criteria",
    legend_title_font=dict(size=14, family="Georgia, bold", color="black"),
    legend_font=dict(size=14, family="Georgia"),
    template='seaborn'
)

fig_correct_order_donut.show()






# Data for the bar chart
bar_data = {
    'Emissions': np.array([0.54586466, 0.27218045, 0.18195489]),
    'Economics': np.array([0.54665687, 0.27112417, 0.18221896]),
    'Safety': np.array([0.54586466, 0.27218045, 0.18195489]),
    'Social': np.array([0.54586466, 0.27218045, 0.18195489])
}

# Overall weightings for the main criteria
overall_weights = np.array([0.19027897, 0.34394734, 0.28204848, 0.18372521])

# Adjust the bar data values based on the overall weightings
adjusted_bar_data = {}
for i, (criterion, values) in enumerate(bar_data.items()):
    adjusted_bar_data[criterion] = values * overall_weights[i]

# Create the adjusted grouped bar chart
fig_adjusted_bar = go.Figure()

# Add bars for each main criterion
for criterion, values in adjusted_bar_data.items():
    for i, value in enumerate(values):
        # Set the x-axis label to be the combination of main and sub criterion
        label = f"{sub_criteria_labels[criterion][i]}"
        fig_adjusted_bar.add_trace(go.Bar(
            x=[label],
            y=[value],
            name=label,
            marker_color=color_map[criterion]
        ))

# Update the layout
fig_adjusted_bar.update_layout(
    title="Adjusted Grouped Bar Chart for Main and Sub Criteria",
    xaxis_title="Criteria",
    yaxis_title="Adjusted Weights",
    barmode='group',
    template='seaborn'
)

#fig_adjusted_bar.show()

# Adjust colors to create variations for the sub-criteria
def adjust_color(color, factor):
    """Lighten or darken a color."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    r = min(255, max(0, r + factor))
    g = min(255, max(0, g + factor))
    b = min(255, max(0, b + factor))
    return f"#{r:02x}{g:02x}{b:02x}"

# Assign specific colors to main criteria based on the colors_plotly list
main_criterion_colors = {
    'Emissions': colors_plotly[0],
    'Economics': colors_plotly[1],
    'Safety': colors_plotly[2],
    'Social': '#d62728'  # New color for the 'Social' main criterion
}

# Create color variations for the sub-criteria based on the main criteria colors
color_variations = {
    criterion: [main_criterion_colors[criterion],
                adjust_color(main_criterion_colors[criterion], 20),
                adjust_color(main_criterion_colors[criterion], -20)]
    for criterion in main_criterion_colors
}

# Create the color-consistent stacked bar chart
fig_consistent_stacked_bar = go.Figure()

# Add bars for each sub-criterion under each main criterion
for criterion, values in adjusted_bar_data.items():
    for i, value in enumerate(values):
        fig_consistent_stacked_bar.add_trace(go.Bar(
            x=[criterion],
            y=[value],
            name=sub_criteria_labels[criterion][i],
            marker_color=color_variations[criterion][i],
            text=sub_criteria_labels[criterion][i],
            textposition='inside',
            textfont=dict(size=16, family="Georgia, bold", color="black")
        ))

# Update the layout for stacking
fig_consistent_stacked_bar.update_layout(
    title="Color-Consistent Stacked Bar Chart for Main and Sub Criteria",
    xaxis_title="Main Criteria",
    yaxis_title="Adjusted Weights",
    barmode='stack',
    template='seaborn',
    legend_title_text="Sub-Criteria",
    legend_title_font=dict(size=20, family="Georgia, bold", color="black"),
    legend_font=dict(size=16, family="Georgia")
)

#fig_consistent_stacked_bar.show()






