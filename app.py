import streamlit as st
import numpy as np
from config import ModelConfig, SourceConfig
from solver import run_simulation
import matplotlib.pyplot as plt

st.title("River Pollutant Transport Simulator")

# Sidebar Controls
st.sidebar.header("Hydraulic Parameters")
velocity = st.sidebar.slider("Flow Velocity (m/s)", 0.1, 2.0, 0.5)

st.sidebar.header("Pollutant Source")
mass = st.sidebar.number_input("Spill Mass (mg)", value=5000)

if st.button("Run Simulation"):
    cfg = ModelConfig()
    cfg.physical.velocity = velocity
    cfg.source.pulse_mass = mass
    
    with st.spinner('Simulating...'):
        result = run_simulation(cfg, verbose=False)
        
        fig, ax = plt.subplots()
        ax.plot(result.x, result.C_w_series[-1])
        ax.set_title("Final Concentration Profile")
        st.pyplot(fig)
        st.success("Done!")
