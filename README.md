
# 📈 Stock Price Prediction 

Welcome to the Stock Price Prediction ! This application allows you to predict stock prices and make informed investment decisions. Below, you'll find instructions on how to use this app effectively.

## Overview:

The Stock Price Prediction App is a powerful tool designed to assist both investors and financial analysts in making informed decisions about stock investments. Developed using Python and Streamlit, this app offers a user-friendly interface for seamless interaction and accurate predictions.

## Features:


📊 **Predictive Analysis:** Utilizes LSTM neural networks to forecast future stock prices based on historical data and technical indicators.


📉 **Technical Indicators:** Computes technical indicators such as Exponential Moving Average (EMA) and Relative Strength Index (RSI) to identify trends and potential buying/selling opportunities.


📈 **Data Visualization:** Presents visually appealing charts and graphs to illustrate historical price trends and predicted future movements.


🛠️ **Customizable Inputs:** Allows users to input various parameters such as ticker symbols and time periods for personalized predictions.


## 🚀 Getting Started


To use the app, follow these steps:

1.Install the required dependencies by running `pip install -r requirements.txt`.


2. Launch the app by executing `streamlit run website_copy.ipnyb.


3. Select a NYSE ticker symbol from the dropdown menu. This will allow you to analyze the stock data for the selected company.


4. Once you've selected a ticker symbol, the app will display the close price history for that stock.


5. Based on the historical data, the app will provide technical recommendations such as Buy, Sell, or Hold.


6. The LSTM model will then be trained to predict future stock prices using the provided data.


7. Finally, you'll receive a recommendation based on the LSTM model's prediction for the next 30 days and stock recommendation.


## ℹ️ Important Information


- **Risk Warning:** The service offered by us may not be suitable for all investors. The stock market is subject to market risk, so please invest carefully.


- **Recommendations:** The app provides technical recommendations based on historical data and LSTM predictions. However, these recommendations should be used as guidance and not as financial advice.


  - **Buy:** Indicates a favorable market condition for buying stocks.

  - **Sell:** Suggests selling existing stocks due to an unfavorable market condition.

  - **Hold:** Advises holding onto existing stocks without making any new purchases or sales.


## 📊 LSTM Model Training

The LSTM model is trained using historical stock data to predict future prices. The training process involves:


- Scaling the data to a specified range.

- Creating input-output pairs for training and testing data.

- Training the LSTM model using the training data.

- Evaluating the model's performance using metrics such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).


## 🔮 Future Predictions


The LSTM model predicts future stock prices for the next 30 days based on the provided data. The predicted prices are displayed along with the average predicted price for the next 30 days.


## 📈 Final Recommendation

Based on the LSTM model's prediction, a final recommendation is provided for the stock. This recommendation is accompanied by an icon indicating whether to Buy, Sell, or Hold the stock.


Contributing:

We welcome contributions from the community to enhance the functionality and accuracy of the Stock price Prediction. Whether it's fixing bugs, adding features, or improving documentation, your contributions are valuable. Fork the repository, make your changes, and submit a pull request. Together, let's make a positive impact on healthcare technology!


📧 Contact Support: 

For inquiries, assistance, or feedback, reach out to us at contact.info.inquiries@gmail.com. We value your input and strive to provide prompt support.


Happy investing! 📈💰
