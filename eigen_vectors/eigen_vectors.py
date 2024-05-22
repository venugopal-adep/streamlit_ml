import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title('Eigenvalues and Eigenvectors Demonstration')
st.write('**Developed by : Venugopal Adep**')

st.write("""
### What is an Eigenvector?
An eigenvector is a vector whose direction remains unchanged when a linear transformation is applied to it. This means that after a transformation represented by a matrix, an eigenvector will only be scaled by a scalar (not rotated). This scalar is known as the eigenvalue associated with the eigenvector.
""")

st.write("""
### How to Use This Application:
1. **Input Matrix**: Use the sidebar to adjust the elements of a 2x2 matrix. 
2. **Observe the Output**: The application calculates the eigenvalues and eigenvectors based on the given matrix.
3. **Interpret the Results**: 
   - **Eigenvalues** show how much the eigenvectors are scaled during the transformation.
   - **Unit Circle vs. Transformed Shape**: Compare how the unit circle is transformed under the matrix to understand the matrix's effects.
   - **Eigenvectors on Plot**: The lines represent the eigenvectors scaled by their corresponding eigenvalues, demonstrating their direction and scaling.
""")

# Input matrix
st.sidebar.header('Matrix Input')
a11 = st.sidebar.number_input('A11', value=1.0)
a12 = st.sidebar.number_input('A12', value=0.0)
a21 = st.sidebar.number_input('A21', value=0.0)
a22 = st.sidebar.number_input('A22', value=1.0)

matrix = np.array([[a11, a12], [a21, a22]])
st.write('Input Matrix:')
st.write(matrix)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
st.write('Eigenvalues:', eigenvalues)
st.write('Eigenvectors:')
st.write(eigenvectors)

# Create unit circle
theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
unit_circle = np.vstack((x, y))

# Apply matrix transformation
transformed = matrix @ unit_circle

# Plotting
fig = go.Figure()

# Original unit circle
fig.add_trace(go.Scatter(x=unit_circle[0, :], y=unit_circle[1, :], mode='lines', name='Unit Circle'))

# Transformed shape
fig.add_trace(go.Scatter(x=transformed[0, :], y=transformed[1, :], mode='lines', name='Transformed'))

# Eigenvectors
for i in range(len(eigenvectors)):
    fig.add_trace(go.Scatter(x=[0, eigenvectors[0, i]*eigenvalues[i]], y=[0, eigenvectors[1, i]*eigenvalues[i]], 
                             mode='lines+markers', name=f'Eigenvec {i+1} (λ={eigenvalues[i]:.2f})'))

fig.update_layout(title='Effect of the Matrix on the Unit Circle',
                  xaxis_title='X', yaxis_title='Y',
                  xaxis=dict(scaleanchor="y", scaleratio=1),
                  yaxis=dict(autorange=True),
                  width=700, height=700)

st.plotly_chart(fig)
