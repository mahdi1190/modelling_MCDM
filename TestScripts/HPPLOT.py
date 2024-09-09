import plotly.graph_objects as go

# Updated data
time_intervals = list(range(10))
carbon_credits_purchased = [160, 177, 150, 144, 155, 151, 162, 159, 168, 150]
carbon_credits_sold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
carbon_credits_held = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
carbon_credits_earned = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

hydrogen_ratio = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
biomass_ratio = [0, 0, 0, 0.7, 0.8, 0.7, 0.7, 0.8, 0.9, 0.8]
natural_gas_ratio = [1, 1, 1, 0.3, 0.2, 0.3, 0.3, 0.2, 0.1, 0.2]


# Carbon Credits Dynamics Plot
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=time_intervals, y=carbon_credits_purchased, mode='lines+markers', name='Purchased', 
                          line=dict(color='black', dash='solid'), marker=dict(symbol='circle')))
fig1.add_trace(go.Scatter(x=time_intervals, y=carbon_credits_sold, mode='lines+markers', name='Sold', 
                          line=dict(color='black', dash='dot'), marker=dict(symbol='square')))
fig1.add_trace(go.Scatter(x=time_intervals, y=carbon_credits_held, mode='lines+markers', name='Banked', 
                          line=dict(color='black', dash='dash'), marker=dict(symbol='diamond')))
fig1.add_trace(go.Scatter(x=time_intervals, y=carbon_credits_earned, mode='lines+markers', name='Earned', 
                          line=dict(color='black', dash='dashdot'), marker=dict(symbol='triangle-up')))
fig1.update_layout(
    #title={'text': 'Carbon Credits Dynamics', 'x':0.5, 'xanchor': 'center'},
    xaxis_title='Time Intervals',
    yaxis_title='No of Credits',
    template='plotly_white',
    font=dict(size=18),
    legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=0.98, xanchor='right', x=0.95)
)
fig1.update_xaxes(title_font=dict(size=22), tickfont=dict(size=18))
fig1.update_yaxes(title_font=dict(size=22), tickfont=dict(size=18))
fig1.update_layout(width=600, height=600)

# Fuel Blending Plot
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=time_intervals, y=hydrogen_ratio, mode='lines+markers', name='Hydrogen', 
                          line=dict(color='black', dash='solid'), marker=dict(symbol='circle')))
fig2.add_trace(go.Scatter(x=time_intervals, y=biomass_ratio, mode='lines+markers', name='Biomass', 
                          line=dict(color='black', dash='dash'), marker=dict(symbol='square')))
fig2.add_trace(go.Scatter(x=time_intervals, y=natural_gas_ratio, mode='lines+markers', name='Natural Gas', 
                          line=dict(color='black', dash='dot'), marker=dict(symbol='diamond')))
fig2.update_layout(
    title={'text': 'Fuel Blending', 'x':0.5, 'xanchor': 'center'},
    xaxis_title='Time Intervals',
    yaxis_title='% Ratio of Fuels',
    template='plotly_white',
    font=dict(size=18),
    legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=0.98, xanchor='right', x=0.95)
)
fig2.update_xaxes(title_font=dict(size=22), tickfont=dict(size=18))
fig2.update_yaxes(title_font=dict(size=22), tickfont=dict(size=18))
fig2.update_layout(width=600, height=600)


fig1.show()
fig2.show()
