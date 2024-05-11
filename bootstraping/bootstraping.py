import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import sem  # Standard Error of the Mean

def generate_data(n_samples=100):
    """ Generate synthetic data """
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=20, size=n_samples)
    return data

def bootstrap_samples(data, n_bootstrap, sample_size):
    """ Perform bootstrap sampling and calculate statistics """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_means.append(np.mean(sample))
    return bootstrap_means

# Streamlit interface setup
st.set_page_config(page_title="Bootstrap Sampling Demo")
st.write("## Bootstrap Sampling Demonstration")
st.write("**Developed by : Venugopal Adep**")
st.markdown("""
This application demonstrates how bootstrap sampling works and how it is used to estimate the distribution of sample means.
""")

# Sidebar for user inputs
n_bootstrap = st.sidebar.slider("Number of Bootstrap Samples", 100, 5000, 1000, help="Adjust to change the number of bootstrap samples generated.")
sample_size = st.sidebar.slider("Sample Size", 1, 100, 50, help="Adjust to change the size of each sample taken from the original data.")

# Generating and sampling data
data = generate_data()
bootstrap_means = bootstrap_samples(data, n_bootstrap, sample_size)

# Plotting the results
fig = px.histogram(bootstrap_means, nbins=50, title="Distribution of Bootstrap Sample Means")
fig.update_layout(xaxis_title='Sample Mean', yaxis_title='Frequency')
st.plotly_chart(fig)

# Displaying statistics
mean_of_means = np.mean(bootstrap_means)
st_dev = np.std(bootstrap_means)
standard_error = sem(bootstrap_means)
st.write(f"Mean of Bootstrap Means: {mean_of_means:.2f}")
st.write(f"Standard Deviation of Bootstrap Means: {st_dev:.2f}")
st.write(f"Standard Error of the Mean: {standard_error:.2f}")

# Explanation of concepts
st.subheader("How to Use This Tool")
st.markdown("""
1. **Adjust the Number of Bootstrap Samples**: This slider allows you to set how many times the sampling process is repeated.
2. **Adjust the Sample Size**: This slider lets you choose how many data points each sample should contain.

After adjustments, the distribution graph and statistics will update automatically.
""")

st.subheader("Explanation of Concepts")
st.markdown("""
Bootstrap sampling is a statistical technique that involves resampling with replacement from a set of data points to estimate statistics on a population. By repeatedly sampling a small subset of data and calculating its mean, we can approximate the distribution of sample means across all possible samples from the original dataset.

**Example**: Imagine we have a dataset of the heights of 100 individuals. By drawing multiple samples of 10 individuals from this dataset (with replacement), and calculating the mean height for each sample, we can create a distribution of sample means. This helps us understand the variability and expected value of the mean height for different samples of this size.
""")
