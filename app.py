import streamlit as st
import numpy as np
from apputil import *

st.write(
'''
# Survival Rate by Class, Sex, and Age Group

'''
)

# Clear question for the results table
st.write("This grouped bar chart shows survival rates across passenger class, gender, and age groups. "
         "Each panel represents a different passenger class, with bars comparing survival rates between males and females across age categories (Child, Teen, Adult, Senior).")
# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Survival by Family Size (Passenger Counts)
'''
)

# Generate and display the figure
st.write("This stacked bar chart shows the absolute number of passengers by family size. "
         "Each bar represents the total passengers in that family size group, with segments showing how many died versus survived.")
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)


st.write(
'''
# Survival Rate by Family Size (Trend Analysis)
'''
)
st.write("This line chart shows how survival rates changed as family size increased. "
         "The trend reveals the optimal family size for survival and shows whether traveling alone or in large groups was more dangerous.")

# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)

# Bonus: Survival rate by age division and class
st.write(
'''
# Survival Rate by Age Division and Class
'''
)
st.write("This bar chart compares survival rates between younger and older passengers within each class. "
         "Age divisions are based on the median age for each class, showing whether relative age within your social class affected survival chances.")
# Generate and display the figure
fig4 = visualize_age_division()
st.plotly_chart(fig4, use_container_width=True)