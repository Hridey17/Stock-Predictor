# importing all the necessary modules
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

start = "2018-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")

stocks = ("AAPL", "GOOGL", "TCS", "MSFT")
selected_stock = st.selectbox("Select Dataset for prediction", stocks)
n_years = st.slider("Years of prediction", 1, 6)
period = n_years*365

@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, start, today) 
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading Data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data... done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)


st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)