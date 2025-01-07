#!/bin/bash
# For Streamlit Web Deployement
# Download the dataset
kaggle datasets download ameencaslam/ddp-v5-runs

# Unzip the models
unzip ddp-v5-runs.zip

# Rename the folder
mv ddp-v5-runs runs

# Cleanup
rm -rf ddp-v5-runs.zip