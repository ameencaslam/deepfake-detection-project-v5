#!/bin/bash
# For Streamlit Web Deployment

# Download the dataset
kaggle datasets download ameencaslam/ddp-v5-runs

# Unzip the models into the 'runs' folder directly
unzip ddp-v5-runs.zip -d runs

# Cleanup
rm -rf ddp-v5-runs.zip
