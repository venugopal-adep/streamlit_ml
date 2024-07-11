import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import sem  # Standard Error of the Mean
import matplotlib.pyplot as plt
import random
import string

# Set page config
st.set_page_config(page_title="Bootstrap Sampling Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6e6;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔄 Bootstrap Sampling Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of bootstrap sampling in estimating population parameters!")

# Helper functions
def generate_data(n_samples=100):
    """ Generate synthetic data """
    return np.random.normal(loc=100, scale=20, size=n_samples)

def bootstrap_samples(data, n_bootstrap, sample_size):
    """ Perform bootstrap sampling and calculate statistics """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_means.append(np.mean(sample))
    return bootstrap_means

def generate_random_sample_names():
    """ Generate random sample names """
    return [f"Sample{random.choice(string.ascii_uppercase)}{random.randint(1, 100)}" for _ in range(3)]

# Sidebar
st.sidebar.header("Sampling Settings")
n_bootstrap = st.sidebar.slider("Number of Bootstrap Samples", 100, 5000, 1000, help="Adjust to change the number of bootstrap samples generated.")
sample_size = st.sidebar.slider("Sample Size", 1, 100, 50, help="Adjust to change the size of each sample taken from the original data.")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_data()
    st.session_state.individual_samples = [np.random.choice(st.session_state.data, size=sample_size, replace=True) for _ in range(3)]
    st.session_state.sample_names = generate_random_sample_names()

# Add Sample button
if st.sidebar.button("Sample"):
    st.session_state.data = generate_data()
    st.session_state.individual_samples = [np.random.choice(st.session_state.data, size=sample_size, replace=True) for _ in range(3)]
    st.session_state.sample_names = generate_random_sample_names()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Solved Example", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Bootstrap Sampling in Action")
    
    # Sampling data
    data = st.session_state.data
    bootstrap_means = bootstrap_samples(data, n_bootstrap, sample_size)
    
    # Plotting the results
    fig = px.histogram(bootstrap_means, nbins=50, title="Distribution of Bootstrap Sample Means")
    fig.update_layout(xaxis_title='Sample Mean', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)
    
    # Displaying statistics
    mean_of_means = np.mean(bootstrap_means)
    st_dev = np.std(bootstrap_means)
    standard_error = sem(bootstrap_means)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean of Bootstrap Means", f"{mean_of_means:.2f}")
    with col2:
        st.metric("Standard Deviation", f"{st_dev:.2f}")
    with col3:
        st.metric("Standard Error", f"{standard_error:.2f}")
    
    # Display 3 individual samples
    st.subheader("Three Individual Samples")
    fig_samples = go.Figure()
    for sample, name in zip(st.session_state.individual_samples, st.session_state.sample_names):
        fig_samples.add_trace(go.Scatter(y=sample, x=np.arange(1, len(sample)+1), mode='markers', name=name))
    fig_samples.update_layout(title='Three Random Samples', xaxis_title='Index', yaxis_title='Value')
    st.plotly_chart(fig_samples, use_container_width=True)

with tab2:
    st.header("Solved Example: Estimating Population Mean")
    
    # Generate a small population
    population = np.random.normal(loc=50, scale=10, size=1000)
    
    # Take a single sample
    single_sample = np.random.choice(population, size=50, replace=False)
    sample_mean = np.mean(single_sample)
    
    # Perform bootstrap
    bootstrap_means = bootstrap_samples(single_sample, 1000, 50)
    
    # Calculate confidence interval
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    
    st.write(f"True Population Mean: {np.mean(population):.2f}")
    st.write(f"Single Sample Mean: {sample_mean:.2f}")
    st.write(f"Bootstrap 95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
    
    # Plot
    fig, ax = plt.subplots()
    ax.hist(bootstrap_means, bins=30, edgecolor='black')
    ax.axvline(np.mean(population), color='r', linestyle='dashed', linewidth=2, label='True Mean')
    ax.axvline(sample_mean, color='g', linestyle='dashed', linewidth=2, label='Sample Mean')
    ax.axvline(ci_lower, color='b', linestyle='dotted', linewidth=2, label='95% CI')
    ax.axvline(ci_upper, color='b', linestyle='dotted', linewidth=2)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of bootstrap sampling?",
            "options": ["To increase sample size", "To estimate sampling distributions", "To reduce data collection costs", "To eliminate outliers"],
            "correct": 1,
            "explanation": "Bootstrap sampling is primarily used to estimate sampling distributions of statistics when the underlying distribution is unknown or complex. It allows us to understand the variability of our estimates without making assumptions about the population distribution."
        },
        {
            "question": "In bootstrap sampling, what does 'sampling with replacement' mean?",
            "options": ["Each sample is returned to the dataset before the next draw", "Samples are taken without returning them to the dataset", "Only unique samples are used", "Samples are replaced with new data"],
            "correct": 0,
            "explanation": "In bootstrap sampling, 'sampling with replacement' means that after each individual sample is drawn, it is put back into the dataset before the next draw. This allows the same data point to potentially be sampled multiple times, mimicking the process of sampling from an infinite population."
        },
        {
            "question": "What does the standard error of the mean represent in bootstrap sampling?",
            "options": ["The average of all sample means", "The variability of the original data", "The variability of the sample means", "The difference between the sample mean and population mean"],
            "correct": 2,
            "explanation": "In bootstrap sampling, the standard error of the mean represents the variability of the sample means. It quantifies how much the sample means tend to deviate from the expected value (i.e., the mean of all sample means). A smaller standard error indicates more precise estimates of the population mean."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
            st.write(f"Explanation: {q['explanation']}")
        st.write("---")

with tab4:
    st.header("Learn More About Bootstrap Sampling")
    st.markdown("""
    Bootstrap sampling is a powerful statistical technique used to estimate the sampling distribution of a statistic by resampling with replacement from the original dataset. It's particularly useful when the underlying distribution is unknown or complex.

    Key benefits of Bootstrap Sampling:
    1. **Non-parametric**: It doesn't require assumptions about the underlying distribution.
    2. **Versatility**: Can be used to estimate various statistics, not just means.
    3. **Confidence Intervals**: Provides a way to calculate confidence intervals for complex statistics.

    How it works:
    1. Start with an original sample of size n.
    2. Resample n items from this sample with replacement.
    3. Calculate the statistic of interest for this resample.
    4. Repeat steps 2-3 many times (typically 1000 or more).
    5. Use the distribution of the calculated statistics to make inferences.

    Applications:
    - Estimating standard errors and confidence intervals
    - Hypothesis testing
    - Model validation in machine learning

    Limitations:
    - Assumes the original sample is representative of the population
    - Can be computationally intensive for large datasets or complex statistics

    Remember, while bootstrap sampling is a powerful tool, it's not a magic solution. It's important to understand its assumptions and limitations when applying it to real-world problems.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the concept of bootstrap sampling. Adjust the settings, click 'Sample' for a new dataset, and explore the different tabs to learn more!")
