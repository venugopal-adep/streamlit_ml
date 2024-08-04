import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import sem
import matplotlib.pyplot as plt
import random
import string

# Set page config
st.set_page_config(page_title="Bootstrap Sampling Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #ffd700;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔄 Bootstrap Sampling Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of bootstrap sampling in estimating population parameters!")

# Helper functions
def generate_data(n_samples=100):
    return np.random.normal(loc=100, scale=20, size=n_samples)

def bootstrap_samples(data, n_bootstrap, sample_size):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_means.append(np.mean(sample))
    return bootstrap_means

def generate_random_sample_names():
    return [f"Sample{random.choice(string.ascii_uppercase)}{random.randint(1, 100)}" for _ in range(3)]

# Sidebar
st.sidebar.header("Sampling Settings")
n_bootstrap = st.sidebar.slider("Number of Bootstrap Samples", 100, 5000, 1000)
sample_size = st.sidebar.slider("Sample Size", 1, 100, 50)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_data()
    st.session_state.individual_samples = [np.random.choice(st.session_state.data, size=sample_size, replace=True) for _ in range(3)]
    st.session_state.sample_names = generate_random_sample_names()

if st.sidebar.button("Sample"):
    st.session_state.data = generate_data()
    st.session_state.individual_samples = [np.random.choice(st.session_state.data, size=sample_size, replace=True) for _ in range(3)]
    st.session_state.sample_names = generate_random_sample_names()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualize", "🧮 Example", "🧠 Quiz"])

with tab1:
    st.header("Understanding Bootstrap Sampling")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Bootstrap Sampling?</h3>
    <p>Bootstrap sampling is like creating multiple versions of your dataset to understand its characteristics better:</p>
    <ul>
        <li>You start with your original data (like a bag of marbles).</li>
        <li>You randomly pick marbles, write down their colors, and put them back.</li>
        <li>You repeat this process many times to create multiple 'new' datasets.</li>
        <li>By analyzing these new datasets, you can estimate properties of your original data more accurately.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Concepts in Bootstrap Sampling</h3>
    <h4>1. Sampling with Replacement</h4>
    <p>This is like picking a marble, noting its color, and putting it back before the next pick. It allows the same data point to be selected multiple times.</p>
    <h4>2. Resampling</h4>
    <p>Creating new samples from your original data. It's like reshuffling your deck of cards multiple times.</p>
    <h4>3. Bootstrap Distribution</h4>
    <p>The distribution of a statistic (like mean or median) calculated from many bootstrap samples. It helps estimate the variability of the statistic.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why Use Bootstrap Sampling?</h3>
    <ul>
        <li><span class="highlight">Estimate Uncertainty:</span> It helps you understand how much your statistics might vary.</li>
        <li><span class="highlight">Non-Parametric:</span> It doesn't assume your data follows a specific distribution.</li>
        <li><span class="highlight">Versatility:</span> It can be used with various types of statistics, not just means.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Bootstrap Sampling in Action")
    
    data = st.session_state.data
    bootstrap_means = bootstrap_samples(data, n_bootstrap, sample_size)
    
    fig = px.histogram(bootstrap_means, nbins=50, title="Distribution of Bootstrap Sample Means")
    fig.update_layout(xaxis_title='Sample Mean', yaxis_title='Frequency', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(marker_color='#8a2be2')
    st.plotly_chart(fig, use_container_width=True)
    
    mean_of_means = np.mean(bootstrap_means)
    st_dev = np.std(bootstrap_means)
    standard_error = sem(bootstrap_means)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean of Bootstrap Means", f"{mean_of_means:.2f}", delta=f"{mean_of_means-np.mean(data):.2f}")
    with col2:
        st.metric("Standard Deviation", f"{st_dev:.2f}")
    with col3:
        st.metric("Standard Error", f"{standard_error:.2f}")
    
    st.markdown("""
    <div style="background-color: #fffacd; padding: 10px; border-radius: 5px;">
    <p><strong>Interpretation:</strong> The histogram shows the distribution of sample means. The narrower this distribution, the more precise our estimate of the true population mean.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Three Individual Samples")
    fig_samples = go.Figure()
    for sample, name in zip(st.session_state.individual_samples, st.session_state.sample_names):
        fig_samples.add_trace(go.Scatter(y=sample, x=np.arange(1, len(sample)+1), mode='markers', name=name))
    fig_samples.update_layout(title='Three Random Samples', xaxis_title='Index', yaxis_title='Value')
    st.plotly_chart(fig_samples, use_container_width=True)

with tab3:
    st.header("Solved Example: Estimating Population Mean")
    
    population = np.random.normal(loc=50, scale=10, size=1000)
    single_sample = np.random.choice(population, size=50, replace=False)
    sample_mean = np.mean(single_sample)
    bootstrap_means = bootstrap_samples(single_sample, 1000, 50)
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    
    st.write(f"True Population Mean: {np.mean(population):.2f}")
    st.write(f"Single Sample Mean: {sample_mean:.2f}")
    st.write(f"Bootstrap 95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
    
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
    
    st.markdown("""
    <div style="background-color: #e0ffff; padding: 15px; border-radius: 8px; margin-top: 20px;">
    <h4>Observations:</h4>
    <ul>
        <li>The bootstrap method provides a range (confidence interval) likely to contain the true population mean.</li>
        <li>This range accounts for the variability in our sampling process.</li>
        <li>Even with a single sample, bootstrap gives us insight into the uncertainty of our estimate.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is bootstrap sampling used for?",
            "options": ["To make the original data larger", "To estimate properties of a population", "To remove outliers from data", "To collect new data"],
            "correct": 1,
            "explanation": "Bootstrap sampling is used to estimate properties (like mean or standard deviation) of a population by resampling from the original data we have."
        },
        {
            "question": "In bootstrap sampling, we sample:",
            "options": ["With replacement", "Without replacement", "From a different dataset", "Only once"],
            "correct": 0,
            "explanation": "In bootstrap sampling, we sample with replacement. This means after we select a data point, we put it back before selecting the next one, allowing the same data point to be chosen multiple times."
        },
        {
            "question": "How many bootstrap samples are typically used?",
            "options": ["Just 1", "Around 10", "Usually 100 or more", "Exactly 50"],
            "correct": 2,
            "explanation": "We typically use a large number of bootstrap samples, usually 100 or more. More samples give us a better estimate of the population properties."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Great job!")
            else:
                st.error("Not quite. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates bootstrap sampling. Adjust the settings, click 'Sample' for new data, and explore the tabs to learn more!")
