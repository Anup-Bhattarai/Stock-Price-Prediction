import streamlit as st

import pandas as pd   #data manipulation and analysis garna ko lagi

import pandas_datareader as pdr  # retrieve financial and economic data from sources

from sklearn.preprocessing import MinMaxScaler  #used for scaling numerical features to a specified range

from keras.models import Sequential  #Sequential is a linear stack of layers

from keras.layers import Dense, LSTM  #Dense is a standard fully connected neural network layer.

import matplotlib.pyplot as plt  #creating static, animated, and interactive visualizations in Python.

from sklearn.metrics import mean_absolute_error #measures the average absolute difference between the predicted values and actual values.

import numpy as np #provides support for arrays, matrices, and mathematical functions to operate on these data structures
# Set page title and icon for stock application
st.set_page_config(page_title="Stock Price Prediction", page_icon=":chart_with_upwards_trend:")


st.markdown("""
    <style>
        .image-container {
            height: 200px; /* Adjust the height as needed */
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .image-container img {
            max-height: 100%;
            width: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='image-container'><img src='https://zipmex.com/static/a024c32a87051954bb94dea9872084da/1bbe7/Bull-vs-Bear-Cover-1.jpg'></div>", unsafe_allow_html=True)

st.markdown("<h1 style='color: black;'>Stock Price Prediction</h1>", unsafe_allow_html=True)
#st.markdown-->method provided by Streamlit for displaying Markdown-formatted text in the app interface.
#unsafe_allow_html=True-->This parameter tells Streamlit to show HTML code that you include in your text
# Function to load NYSE tickers from a CSV file
def load_nyse_file_from_github(url):
    return pd.read_csv(url)['Ticker']

# Fetch NYSE tickers
url = 'https://raw.githubusercontent.com/dhhagan/stocks/master/scripts/stock_info.csv'
nyse_tickers = load_nyse_file_from_github(url)

# Function to fetch and process data
def fetch_and_process_data(ticker, api_key):
    df = pdr.get_data_tiingo(ticker, api_key=api_key)
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    if df.isnull().sum().any():
        df.dropna(inplace=True)
    return df

# Get NYSE ticker symbols and corresponding full names
def load_nyse_tickers_and_names(url):
    df = pd.read_csv(url)
    return dict(zip(df['Ticker'], df['Name']))

# Get NYSE tickers and their full names
nyse_tickers_with_names = load_nyse_tickers_and_names(url)

# Enhanced Ticker Symbol Input with full names
ticker_symbol = st.selectbox("Enter the NYSE ticker symbol to analyze", [''] + list(nyse_tickers_with_names.keys()), format_func=lambda x: f"{x} - {nyse_tickers_with_names[x]}" if x else "")

# API key for data retrieval
api_key = '83928d2d74729aa4d2c32ca3eb6f428356745d44'

# process data if a ticker symbol is selected
if ticker_symbol and ticker_symbol != '':
    df = fetch_and_process_data(ticker_symbol, api_key=api_key)

elif ticker_symbol == '':
    st.warning("Please select a ticker symbol to analyze.")

# Check if the ticker symbol is in the NYSE tickers list
if ticker_symbol in nyse_tickers.values:
    df = fetch_and_process_data(ticker_symbol,api_key)
    # Visualizations
    st.subheader('Close price History for ' + ticker_symbol)
    fig = plt.figure(figsize=(20, 15))
    plt.title(f'Close Price History ', fontsize=16)
    plt.plot(df['close'])
    plt.ylabel('Close Price in $', fontsize=16)
    st.pyplot(fig)
    ticker_symbols = [ticker_symbol]
    results = pd.DataFrame(columns=['Ticker', 'EMA', 'RSI', 'Technical Recommendation'])

    for ticker in ticker_symbols:
        try:


            closing_price = df['close'].iloc[-1]


            
            # Define parameters
            num_observations = 20
            multiplier = 2 / (num_observations + 1)

            # Calculate the Average for the last 20 days
            sma_last_20_days = df['close'].iloc[-20:].mean()


            # Extract the last SMA value (EMA for the previous day)
            ema_previous_day = sma_last_20_days

            # Extract the closing price of the last day
            closing_price = df['close'].iloc[-1]

            # Calculate the current EMA
            ema = closing_price * multiplier + ema_previous_day * (1 - multiplier)





            # Extract the last 14 days of closing prices
            closing_prices_last_14_days = df['close'].tail(14)

            # Calculate delta, gain, and loss for the last 14 days
            delta = closing_prices_last_14_days.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and average loss for the last 14 days
            avg_gain = gain.mean()
            avg_loss = loss.mean()

            # Calculate RSI for the last 14 days
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))


           

            

            if (closing_price < ema) and (rsi<30):
                recommendation = 'Buy'
                recon_index = 1


            elif (closing_price > ema) and (rsi>70):
                recommendation = 'Sell'
                recon_index =-1


            else:
                recommendation = 'Hold'
                recon_index= 0


            result = pd.DataFrame({'Ticker': [ticker],
                                   'EMA': [ema],
                                   'RSI': [rsi],
                                   'Technical Recommendation': [recommendation]})
            results = pd.concat([results, result], ignore_index=True)
        except Exception as e:
            st.write(f"Error fetching data for {ticker}: {e}")

    st.write(results.to_markdown(index=False))
    st.write("")
    st.markdown("""
        <style>
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }

        .blink {
            animation: blink 1s linear infinite;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<h6 class='blink' style='color: black;'>Training LSTM Model...........</h6>", unsafe_allow_html=True)

    close = df['close']
    ds = close.values
    ds = np.array(ds).reshape(-1, 1) #This line first converts the dataset ds into a NumPy array using np.array()
    normalizer = MinMaxScaler(feature_range=(0, 1)) # This scaler will transform the data to a specified range
    ds_scaled = normalizer.fit_transform(ds)


    train_size = int(len(ds_scaled) * 0.65) #This line calculates the size of the training set.
    # It takes the length of the scaled dataset ds_scaled and multiplies it by 0.65 (65%)

    test_size = len(ds_scaled) - train_size
    #This line calculates the size of the testing set.
    # #It subtracts the size of the training set (train_size) from the total length of the scaled dataset ds_scaled.

    ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]
    #This line actually performs the split. It slices the ds_scaled array into two subsets:
    # ds_train for training data and ds_test for testing data.


    def create_dataset(dataset, step):
        X, Y = [], []
        for i in range(len(dataset) - step - 1):
            a = dataset[i:(i + step), 0]
            X.append(a)
            Y.append(dataset[i + step, 0])
        return np.array(X), np.array(Y)
    #We define a "step", which tells us how many time steps into the past we want to look when making predictions.

    #The function then creates two lists, X and Y, which will hold our input features and target values, respectively.
    #Next, it goes through the dataset, stopping step + 1 steps before the end.
    # This ensures we have enough data for both input and target at each step.
    #For each step, it creates an input sequence (X) consisting of step time steps from the dataset.
    #It also grabs the next value in the series, which becomes the target (Y).
    #These input-output pairs are then added to the X and Y lists.
    #Finally, the function converts X and Y into NumPy arrays and returns them


    time_stamp = 60
    X_train, y_train = create_dataset(ds_train, time_stamp)
    X_test, y_test = create_dataset(ds_test, time_stamp)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #X_train, y_train = create_dataset(ds_train, time_stamp): Generates input-output pairs for training data,
    # considering the specified number of past time steps.
    #X_test, y_test = create_dataset(ds_test, time_stamp): Generates input-output pairs for testing data,
    # using the same number of past time steps.
    #X_train.shape[0]: This retrieves the number of rows (samples) in the X_train array.
    #X_train.shape[1]: This retrieves the number of columns (time steps) in the X_train array.
    #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1): Reshapes the training input data to fit the expected input shape
    #X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1): Reshapes the testing input data to match the format of the training data.

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=64, verbose=1)
    #batch_size=64: Think of this as how many questions the model tries to answer before checking if it's getting better.
    # #It looks at 64 questions at a time before adjusting its understanding.
    #A value of 1 means that training progress will be displayed in detail, including the loss and metrics at each epoch.
    #If verbose is set to 2, it means that the training progress will be displayed, but it will be less detailed compared to verbose=1

    loss = model.history.history['loss']
    fig = plt.figure(figsize=(20, 15))
    plt.plot(loss)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.show()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    # Calculate Mean Absolute Error (MAE) on test data
    mae = mean_absolute_error(normalizer.inverse_transform(y_test.reshape(-1, 1)), test_predict)
    print(f"Mean Absolute Error (MAE) on Test Data: {mae}")



    # Calculate Mean Absolute Percentage Error (MAPE)
    def calculate_mape(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape







    # Calculate MAPE for test data
    mape_test = calculate_mape(normalizer.inverse_transform(y_test.reshape(-1, 1)), test_predict)
    print(f"Mean Absolute Percentage Error (MAPE) on Test Data: {mape_test:.2f}%")






    # Define the metrics
    metrics_data = {
        "Metric": ["Mean Absolute Percentage Error (MAPE)",
                   "Mean Absolute Error (MAE)"],
        "Value": [mape_test, mae]
    }

    # Create a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Render the DataFrame using Streamlit
    st.write(metrics_df)


    # Create label for test data
    test_labels = np.vstack((normalizer.inverse_transform(y_test.reshape(-1, 1)), test_predict))

    # Plotting
    fig = plt.figure(figsize=(15, 7.5))
    plt.ylabel('Close Price in ($)', fontsize=18)
    plt.xlabel('Time', fontsize=18)

    # Plotting actual close prices (blue line)
    plt.plot(normalizer.inverse_transform(ds_scaled), label='Actual Close Prices')

    # Plotting predicted close prices (orange line)
    plt.plot(test_labels, label='Predicted Close Prices')

    # Adding legend
    plt.legend() #This helps viewers understand the data being presented.


    # Display the plot
    st.pyplot(fig)

    future_input = ds_test[-time_stamp:]#This line selects the last time_stamp data points from the ds_test time series dataset,
    # indicating the number of past time steps considered for prediction.

    future_input = future_input.reshape(1, -1)
    #future_input = future_input.reshape(1, -1): Reshapes the input data for compatibility with the model's expected format.

    tmp_input = list(future_input[0])
    #tmp_input = list(future_input[0]): Converts the reshaped input data into a list for easier handling.

    future_output = []
    #future_output = []: Initializes an empty list to store future predictions.
    n_steps = time_stamp

    for i in range(30):#for i in range(30): This loop runs 30 times, indicating that the model will make 30 future predictions.
        if len(tmp_input) >= n_steps:#Checks if the length of tmp_input is greater than or equal to n_steps
            future_input = np.array(tmp_input[-n_steps:])#Selects the last n_steps values from tmp_input.
            future_input = future_input.reshape(1, -1)#Reshapes the selected values to match the model's input shape.
            future_input = future_input.reshape((1, n_steps, 1))
            forecast = model.predict(future_input, verbose=0)#Predicts the next value (forecast) using the model.
            tmp_input.extend(forecast[0].tolist())#Extends tmp_input with the predicted value
            future_output.extend(forecast.tolist())#Extends future_output with the predicted value.
        else:
            future_input = future_input.reshape((1, len(tmp_input), 1)) #Reshapes future_input to match the model's input shape.
            forecast = model.predict(future_input, verbose=0)
            tmp_input.extend(forecast[0].tolist())
            future_output.extend(forecast.tolist())
            #In summary, this loop iterates 30 times to predict future values using the model.
            # #It checks if there are enough past values (tmp_input) to make predictions.
            # #If there are, it selects the past values, reshapes them, predicts the next value,
            # #and appends it to tmp_input and future_output. If there aren't enough past values,
            # it reshapes the available past values, predicts the next value, and appends it to tmp_input and future_output.

    future_output = normalizer.inverse_transform(future_output)
    final_graph = normalizer.inverse_transform(ds_scaled).tolist()
    final_graph.extend(future_output)
    average_predicted_price = np.mean(final_graph[-30:])


    last_close_price = df['close'].iloc[-1]
    if average_predicted_price-10 > last_close_price:
        final_recommendation = 'Buy'
        recommend_index=1
    elif average_predicted_price+10 < last_close_price:
        final_recommendation = 'Sell'
        recommend_index=-1
    else:
        final_recommendation = 'Hold'
        recommend_index=0





    fig=plt.figure(figsize=(20, 15))
    plt.plot(final_graph)
    plt.ylabel('Price', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.axhline(y=final_graph[-1][0], color='red', linestyle=':', label=f'NEXT 30D: {round(float(final_graph[-1][0]), 2)}')
    plt.legend()
    st.pyplot(fig)

    st.write(f"Average Predicted Price for the Next 30 Days: ${round(average_predicted_price, 2)}")

    # Define color based on LSTM recommendation
    recommendation_color = "green" if final_recommendation == "Buy" else "red" if final_recommendation == "Sell" else "gray"

    # Define text color based on background
    text_color = "black" if recommendation_color == "gray" else "white"

    # Write recommendation with colored box
    st.markdown(
        f'<div style="background-color:{recommendation_color}; padding: 8px 12px; border-radius: 8px; display: inline-block;">'
        f'<span style="color:{text_color}; font-size: 18px;">LSTM Recommendation: {final_recommendation}</span></div>',
        unsafe_allow_html=True
    )

    # Define recommendation text and icon based on the final recommendation
    if recommend_index == recon_index:
        if recon_index == 0:
            recommendation_text = "Hold"
            icon = "⏸️"  # Pause icon
        elif recon_index == 1:
            recommendation_text = "Buy"
            icon = "🟢⬆️"  # Green upward arrow icon for Buy
        else:
            recommendation_text = "Sell"
            icon = "🔴⬇️"  # Red downward arrow icon for Sell
    else:
        recommendation_text = "Hold"
        icon = "⏸️"  # Pause icon

    # Display recommendation with styled text and icon
    st.markdown(f"<h3 style='color:#3366ff;'>Final Recommendation: {icon} {recommendation_text}</h3>",
                unsafe_allow_html=True)
else:
    st.write(f"Ticker symbol {ticker_symbol} not found in NYSE list.")
# Define the CSS style with animation
css = """
@keyframes slidein {
    from {
        margin-left: 100%;
        width: 200%; 
    }

    to {
        margin-left: 0%;
        width: 400%;
    }
}

.slide-in {
    animation: slidein 7s infinite linear;
}
"""

# Display the CSS style
st.write("<style>{}</style>".format(css), unsafe_allow_html=True)

st.markdown("<div class='slide-in'>The service offered by us may not be suitable for all investors,Stock market is subject to market risk please invest carefully.</div>", unsafe_allow_html=True)