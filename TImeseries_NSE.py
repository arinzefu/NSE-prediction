

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf



df = pd.read_csv('NSE All Share Historical Data.csv')

df.shape
df.head()


df.describe()

df.isnull().sum()

df.replace(',', '', regex=True)


df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')


df['Date']


# In[75]:


print(df.dtypes)


# In[76]:


df.head()


# In[77]:


df['Vol.'].fillna(method='bfill', inplace=True)

df['Vol.'] = df['Vol.'].str.replace('M', 'e6').str.replace('B', 'e9').astype(float)


# In[78]:


df.drop(columns=['Change %', 'Volume'], inplace=True)

# List of columns to convert to float
columns_to_convert = ['Price', 'Open', 'High', 'Low',]

# Convert the specified columns to float
df[columns_to_convert] = df[columns_to_convert].apply(lambda x: pd.to_numeric(x.str.replace(',', '', regex=True), errors='coerce', downcast='float'))


df.dtypes

df.isnull().sum()


df['Vol.'].fillna(method='ffill', inplace=True)

df.isnull().sum()

sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Graph of NSE Share Historical Data')
plt.show()

sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Vol.', color='red')
plt.xlabel('Date')
plt.ylabel('Volume of Share')
plt.title('Graph of NSE Share Purchased Each Day')
plt.show()

top_20_prices = df.sort_values(by='Price', ascending=False).head(20)
print(top_20_prices[['Date', 'Price', 'Vol.']])


top_20_Volume = df.sort_values(by='Vol.', ascending=False).head(20)
print(top_20_Volume[['Date', 'Price', 'Vol.']])

bottom_20_prices = df.sort_values(by='Price', ascending=True).head(20)
print(bottom_20_prices[['Date', 'Price', 'Vol.']])


bottom_20_Volume = df.sort_values(by='Vol.', ascending=True).head(20)
print(bottom_20_Volume[['Date', 'Price', 'Vol.']])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Resample the data to yearly frequency, calculating the mean for 'Price' and 'Vol.'
yearly_avg = df.resample('Y').mean()

# Reset the index to make 'Date' a regular column
yearly_avg.reset_index(inplace=True)

# Print the table
print(yearly_avg[['Date', 'Price', 'Vol.']])

df.head()

print(df)


sns.set_style('darkgrid')
plt.figure(figsize=(12,6))


sns.lineplot(data=df, x='Date', y='Price', color='red', label='Price')

sns.lineplot(data=df, x='Date', y='High', color='blue', label='High')

sns.lineplot(data=df, x='Date', y='Low', color='green', label='Low')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('NSE Share Historical Data')
plt.legend()
plt.show()


df.drop(columns=['Open', 'High', 'Low', 'Vol.'], inplace=True)


# Define the split index values
split_index1 = '2021-01-01'
split_index2 = '2022-07-01'

# Convert the index to a DateTimeIndex (if it's not already)
df.index = pd.to_datetime(df.index)

# Split the data into training, validation, and test sets
train_data = df[df.index < split_index1]
val_data = df[(df.index >= split_index1) & (df.index < split_index2)]
test_data = df[df.index >= split_index2]


# Plot the entire dataset
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Full Dataset', color='blue')

# Plot the training, validation, and test sets
plt.scatter(train_data.index, train_data['Price'], label='Training Set', color='green', marker='o')
plt.scatter(val_data.index, val_data['Price'], label='Validation Set', color='orange', marker='o')
plt.scatter(test_data.index, test_data['Price'], label='Test Set', color='red', marker='o')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Dataset Split: Training, Validation, and Test Sets')
plt.legend()

# Show the plot
plt.show()

train_data.shape


from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the training, validation, and test data 
train_scaled = scaler.fit_transform(train_data['Price'].values.reshape(-1, 1))
val_scaled = scaler.transform(val_data['Price'].values.reshape(-1, 1))
test_scaled = scaler.transform(test_data['Price'].values.reshape(-1, 1))

train_scaled

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 20
X_train, y_train = create_dataset(train_scaled, time_step)
X_val, y_val = create_dataset(val_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)

X_train.shape

y_train.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

features = 1
# Define the model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, features)))
model.add(Dropout(0.1))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

from tensorflow.keras.optimizers import Adam

# Define the learning rate
learning_rate = 0.001

# Compile the model 
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=100,batch_size=64,verbose=1, callbacks=[early_stopping])

test_loss, test_mae = model.evaluate(X_train, y_train)
print('Test Loss:', test_loss)
print('Test MAE:', test_mae)


# # Graph of the Predicted Train Price vs the Actual Price

Train_predictions = model.predict(X_train)

train_prediction=scaler.inverse_transform(Train_predictions)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
plt.figure(figsize=(12,6))
plt.plot(y_train_original[::-1], label='Actual')
plt.plot(train_prediction[::-1], label='Predicted')
plt.title('Actual vs Predicted Train set')
plt.legend()
plt.show()


Val_predictions = model.predict(X_val)

Val_prediction=scaler.inverse_transform(Val_predictions)
y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
plt.figure(figsize=(12,6))
plt.plot(y_val_original[::-1], label='Actual')
plt.plot(Val_prediction[::-1], label='Predicted')
plt.title('Actual vs Predicted Validation set')
plt.legend()
plt.show()

test_predictions = model.predict(X_test)

Test_prediction=scaler.inverse_transform(test_predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.figure(figsize=(12,6))
plt.plot(y_test_original[::-1], label='Actual')
plt.plot(Test_prediction[::-1], label='Predicted')
plt.title('Actual vs Predicted Test set')
plt.legend()
plt.show()

# # Save the model

model.save('NSE_model_1.keras')


# # Now only training with the last 1 year of the Dataset

df.head()

# Define the split index values
split_index1 = '2022-01-10'
split_index2 = '2023-05-27'

# Split the data into training and test sets
new_train = df[(df.index > split_index1) & (df.index < split_index2)]
new_test = df[df.index >= split_index2]

# Plot the dataset
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Full Dataset', color='blue')

# Plot the training and test sets based on the new split dates
plt.scatter(new_train.index, new_train['Price'], label='Training Set', color='green', marker='o')
plt.scatter(new_test.index, new_test['Price'], label='Test Set', color='red', marker='o')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Dataset Split: Training and Test Sets')
plt.legend()

# Show the plot
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the training, validation, and test data 
train_scaled_1 = scaler.fit_transform(new_train.values.reshape(-1, 1))

test_scaled_1 = scaler.transform(new_test.values.reshape(-1, 1))

time_step = 10
X_train, y_train = create_dataset(train_scaled_1, time_step)

X_test, y_test = create_dataset(test_scaled_1, time_step)

# Define the model architecture  
model_2 = Sequential()
model_2.add(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)))
model_2.add(Dropout(0.1))
model_2.add(LSTM(units=100, return_sequences=True))
model_2.add(Dropout(0.1))
model_2.add(LSTM(units=100)) 
model_2.add(Dropout(0.1))
model_2.add(Dense(units=1, activation='linear', kernel_regularizer='l2'))

model_2.summary()

# Define the learning rate
learning_rate = 0.01

# Compile the model 
model_2.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history_2 = model_2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=32,verbose=1, callbacks=[early_stopping])

Train_predictions = model_2.predict(X_train)

train_prediction=scaler.inverse_transform(Train_predictions)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
plt.figure(figsize=(12,6))
plt.plot(y_train_original[::-1], label='Actual')
plt.plot(train_prediction[::-1], label='Predicted')
plt.title('Actual vs Predicted Train set')
plt.legend()
plt.show()

test_predictions = model_2.predict(X_test)

Test_prediction=scaler.inverse_transform(test_predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.figure(figsize=(12,6))
plt.plot(y_test_original[::-1], label='Actual')
plt.plot(Test_prediction[::-1], label='Predicted')
plt.title('Actual vs Predicted Test set')
plt.legend()
plt.show()

model_2.save('NSE_model_2.keras')