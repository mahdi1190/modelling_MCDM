import plotly.graph_objects as go
import numpy as np

# Updated data for the new plot
time_intervals = list(range(10))

# Creating non-linear data for max working capacity and temperature lift with distinct trajectories
# Using an exponential decay function to create a fast start that slows down
max_working_capacity = [1 + (2 * (1 - np.exp(-0.3 * t))) for t in time_intervals]  # in MW
temperature_lift = [200 + (200 * (1 - np.exp(-0.5 * t))) for t in time_intervals]  # in degrees Celsius

# Heat Pump Dynamics Plot with Dual Y-Axis
fig = go.Figure()

# Add the maximum working capacity trace
fig.add_trace(go.Scatter(
    x=time_intervals, y=max_working_capacity, mode='lines+markers', name='Max Working Capacity (MW)', 
    line=dict(color='black', dash='solid'), marker=dict(symbol='circle')
))

# Add the temperature lift trace with a secondary y-axis
fig.add_trace(go.Scatter(
    x=time_intervals, y=temperature_lift, mode='lines+markers', name='Max Working Temperature (°C)', 
    line=dict(color='black', dash='dash'), marker=dict(symbol='square'), yaxis='y2'
))

# Add a vertical line to indicate the investment at the 4th time interval
fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=4,
        x1=4,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="black", width=2, dash="dot")
    )
)

# Update the layout to include the secondary y-axis, position the legend at the top center, and add the annotation
fig.update_layout(
    xaxis_title='Time Intervals',
    yaxis=dict(
        title='Max Working Capacity (MW)',
        titlefont=dict(size=22),
        tickfont=dict(size=18),
        range=[0, 3.5]
    ),
    yaxis2=dict(
        title='Max Working Temperature (°C)',
        titlefont=dict(size=22),
        tickfont=dict(size=18),
        overlaying='y',
        side='right',
        range=[180, 420]
    ),
    template='plotly_white',
    font=dict(size=18),
    legend=dict(
        font=dict(size=14),
        orientation='h',
        yanchor='top',
        y=1.15,  # Position legend above the plot
        xanchor='center',
        x=0.5
    ),
    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins to fit legend
    width=600,
    height=600,
    annotations=[
        go.layout.Annotation(
            x=4,  # x-coordinate for the annotation
            y=max_working_capacity[4],  # y-coordinate for the annotation
            xref="x",
            yref="y",
            text="Investment Made",
            showarrow=True,
            arrowhead=2,
            ax=100,
            ay=10,
            font=dict(size=17)
        )
    ]
)
fig.update_xaxes(title_font=dict(size=22), tickfont=dict(size=18))

fig.show()
