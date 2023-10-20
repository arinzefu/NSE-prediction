Nigerian Stock Exchange Price Prediction Project

This Python script constitutes a comprehensive data analysis and machine learning project centered on historical stock market data from the NSE (Nigerian Stock Exchange). The project encompasses the following stages:

1. Data Acquisition and Preprocessing:
   - The script begins by importing essential libraries.
   - It reads historical stock market data from a CSV file into a pandas DataFrame.
   - Data cleaning and preprocessing tasks are performed.

2. Data Visualization:
   - Utilizing Matplotlib and Seaborn, the script creates various graphs and visual insights to better understand the dataset.

3. Machine Learning for Stock Price Prediction:
   - The script prepares the data by scaling and generating input sequences for an LSTM neural network.
   - It defines a deep learning model using TensorFlow for stock price forecasting.
   - The model is configured and trained on both training and validation datasets, with early stopping to optimize performance and prevent overfitting.

4. Model Evaluation and Visualization:
   - The script evaluates the trained model's performance by calculating loss and mean absolute error.
   - It provides visualizations of the model's predictions on training and validation datasets, enabling users to assess prediction quality.

As the initial model faced challenges predicting recent trends, the script introduces a new model using data from 2022:

5. Data Split for Model Re-Training:
   - The dataset is split into a new training and test set, focusing on the most recent year of available data.

6. Second Machine Learning Model:
   - A new LSTM-based model is defined and compiled with a different architecture for stock price prediction.
   - This model is trained and evaluated using the new training and test datasets.

The second model demonstrated improved predictive capabilities, although this year's data has outperformed the model's predictions.