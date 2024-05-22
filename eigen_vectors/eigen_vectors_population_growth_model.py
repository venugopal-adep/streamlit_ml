import streamlit as st
import numpy as np
import plotly.graph_objects as go

def plot_vector(v, color='blue'):
    fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode='lines', line=dict(color=color, width=2)))

def update_plot():
    A = np.array([[a11, a12], [a21, a22]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig.data = []
    plot_vector(eigenvectors[:, 0], color='red')
    plot_vector(eigenvectors[:, 1], color='green')
    
    for i in range(num_points):
        v = np.array([np.cos(i * 2 * np.pi / num_points), np.sin(i * 2 * np.pi / num_points)])
        transformed_v = A.dot(v)
        plot_vector(v)
        plot_vector(transformed_v, color='purple')
    
    fig.update_layout(title='Eigenvectors and Transformed Vectors', 
                      xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]))
    st.plotly_chart(fig)
    
    st.write('Eigenvalues:', eigenvalues)
    st.write('Eigenvectors:', eigenvectors)

st.title('Eigen Vectors - Population Growth Model')
st.write('**Developed by : Venugopal Adep**')
st.write("**:red[Please note that this is Under Development]**")

st.sidebar.markdown('## Matrix Parameters')
a11 = st.sidebar.slider('Birth Rate (a11)', -5.0, 5.0, 1.0, 0.1)
a12 = st.sidebar.slider('Migration Rate (a12)', -5.0, 5.0, 0.0, 0.1)
a21 = st.sidebar.slider('Death Rate (a21)', -5.0, 5.0, 0.0, 0.1)
a22 = st.sidebar.slider('Survival Rate (a22)', -5.0, 5.0, 1.0, 0.1)
num_points = st.sidebar.slider('Number of Vectors', 1, 100, 20, 1)

st.markdown('## Explanation')
st.write('''
This application demonstrates the concept of eigenvalues and eigenvectors in the context of a population growth model. 
The matrix represents the transition rates between two age groups (young and old) in a population.

- Birth Rate (a11): The rate at which the young population reproduces.
- Migration Rate (a12): The rate at which individuals migrate from the old age group to the young age group.
- Death Rate (a21): The rate at which individuals in the young age group die.
- Survival Rate (a22): The rate at which individuals in the old age group survive.

The eigenvectors (red and green arrows) represent the stable age distribution of the population. They indicate the proportions of the young and old age groups in the long run.

The blue arrows represent different initial age distributions, and the purple arrows show how these distributions change over time when multiplied by the transition matrix.
''')

fig = go.Figure()
update_plot()
