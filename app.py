import streamlit as st
import numpy as np
from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write(
'''
# Titanic Visualization 1

'''
)

# Clear question for the results table
st.write("The grouped Bar chart has been generated using Plotly Express. "
         "The chart visualizes the survival rate of Titanic passengers based on their passenger class")
# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2
'''
)


# Generate and display the figure
st.write("The bar chart below visualizes the number of families on the Titanic based on their family size. "
         "The chart is stacked to show the number of survivors and non-survivors within each family size category.")
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)


st.write(
'''
# Titanic Visualization Bonus
'''
)
st.write("The line chart below visualizes the survival rate of Titanic passengers based on their age division and passenger class. "
         "The chart helps to understand how age and class influenced the chances of survival during the Titanic disaster.")

# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)

# Bonus: Survival rate by age division and class
st.write(
'''
# Titanic Age Division Analysis
'''
)
st.write("The bar chart below visualizes the survival rate of Titanic passengers based on their age division (younger or older than 18) and passenger class. "
         "The chart helps to understand how age and class influenced the chances of survival during the Titanic disaster.")
# Generate and display the figure
fig4 = visualize_age_division()
st.plotly_chart(fig4, use_container_width=True)