import streamlit as st 
import yfinance as yf
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from htmlTemplates import css

load_dotenv()  # Load API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

def init_ses_states():
    st.session_state.setdefault('chat_history', [])

def relative_returns(df):
    rel = df.pct_change()
    cumret = ((1 + rel).cumprod() - 1).fillna(0)
    return cumret

def ask_openai(prompt, stock_tickers=None):
    """ General AI query function """
    if not openai_api_key:
        return "OpenAI API key is missing."

    client = openai.OpenAI(api_key=openai_api_key)

    # If stock tickers are provided, format them into the prompt
    if stock_tickers:
        prompt = f"Stocks: {', '.join(stock_tickers)}\nUser Query: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a financial market assistant providing stock insights, trends, and analysis."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def get_stock_summary(stock_tickers, start, end):
    """ AI generates a stock performance summary based on price data """
    if not stock_tickers:
        return "⚠️ No stocks selected. Please choose at least one stock before proceeding."

    df = yf.download(stock_tickers, start, end, auto_adjust=True)
    
    if df.empty:
        return "No stock data found for the selected tickers."

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs('Close', axis=1, level=0, drop_level=True)
    elif 'Close' in df.columns:
        df = df[['Close']]
    else:
        return "Stock data does not contain 'Close'."

    # Generate summary
    summary = f"Stock Performance Summary for {', '.join(stock_tickers)}:\n"
    for ticker in df.columns:
        latest_price = df[ticker].dropna().iloc[-1]
        start_price = df[ticker].dropna().iloc[0]
        change = ((latest_price - start_price) / start_price) * 100
        summary += f"\n- {ticker}: ${latest_price:.2f} (Change: {change:.2f}%)"

    # Send data to AI for analysis
    ai_prompt = f"""
    Analyze the following stock data: {summary}
    Provide a natural language summary of the stock trends, market movements, and potential factors affecting performance.
    """
    
    return ask_openai(ai_prompt)

def get_risks_and_opportunities(stock_tickers):
    """ AI provides investment risks and opportunities analysis """
    if not stock_tickers:
        return "⚠️ No stocks selected. Please choose at least one stock before proceeding."

    ai_prompt = f"""
    Analyze the risks and opportunities for the following stocks: {', '.join(stock_tickers)}.
    
    Consider:
    - Market conditions
    - Recent earnings reports
    - Competitive landscape
    - Macroeconomic factors
    - Emerging trends in the industry
    
    Provide an overview of the biggest risks and potential opportunities for each stock.
    """
    
    return ask_openai(ai_prompt)

def main():
    st.set_page_config(page_title="Stock Price AI Bot", page_icon=":chart:")
    st.write(css, unsafe_allow_html=True)
    init_ses_states()
    st.title("Stock Price AI Bot")
    st.caption("Visualizations & AI Insights for Stocks")

    with st.sidebar:
        with st.expander("Settings", expanded=True):
            asset_tickers = sorted(['DOW','NVDA','TSL','GOOGL','AMZN','AI','NIO','LCID','F','LYFY','AAPL', 'MSFT', 'BTC-USD', 'ETH-USD'])
            asset_dropdown = st.multiselect('Pick Assets:', asset_tickers)

            metric_tickers = ['Adj. Close', 'Relative Returns']
            metric_dropdown = st.selectbox("Metric", metric_tickers)

            viz_tickers = ['Line Chart', 'Area Chart']
            viz_dropdown = st.multiselect("Pick Charts:", viz_tickers)

            start = st.date_input('Start', value=pd.to_datetime('2023-01-01'))
            end = st.date_input('End', value=pd.to_datetime('today'))

    if len(asset_dropdown) == 0:
        st.warning("Please select at least one stock.")
    else:
        df = yf.download(asset_dropdown, start, end, auto_adjust=True)
    
        if df.empty:
            st.error(f"No data available for {asset_dropdown}. Please check the ticker symbols.")
            st.stop()
        
        st.write("Downloaded Data Preview:", df.head())

        # Select Adjusted Close price
        if 'Adj Close' in df.columns:
            df = df['Adj Close']
        elif 'Close' in df.columns:
            df = df['Close']
        else:
            st.error("Stock data does not contain 'Adj Close' or 'Close'. Check ticker symbols.")
            st.stop()

    if metric_dropdown == 'Relative Returns':
        df = relative_returns(df)
    
    if len(viz_dropdown) > 0:
        with st.expander(f"Data Visualizations for {metric_dropdown} of {asset_dropdown}", expanded=True):
            if "Line Chart" in viz_dropdown:
                st.subheader("Line Chart")
                st.line_chart(df)
            if "Area Chart" in viz_dropdown:
                st.subheader("Area Chart")
                st.area_chart(df)

    st.subheader("AI Stock Insights")

    # AI Stock Performance Summary
    if st.button("📊 Get AI Stock Performance Summary"):
        if not asset_dropdown:
            st.warning("⚠️ Please select at least one stock before requesting a summary.")
        else:
            summary = get_stock_summary(asset_dropdown, start, end)
            st.write("### AI Performance Summary:")
            st.write(summary)

    # AI Risks & Opportunities Analysis
    if st.button("💡 Get AI Risks & Opportunities Analysis"):
        if not asset_dropdown:
            st.warning("⚠️ Please select at least one stock before requesting risk & opportunity analysis.")
        else:
            risk_opportunities = get_risks_and_opportunities(asset_dropdown)
            st.write("### AI Risks & Opportunities:")
            st.write(risk_opportunities)

    # AI Chatbot for Stock Queries
    user_query = st.text_input("Ask AI about selected stocks:")
    if st.button("Ask AI"):
        if not asset_dropdown:
            st.warning("⚠️ Please select at least one stock before asking AI.")
        elif not user_query:
            st.warning("⚠️ Please enter a question before asking AI.")
        else:
            ai_response = ask_openai(user_query, asset_dropdown)
            st.write("### AI Response:")
            st.write(ai_response)

if __name__ == '__main__':
    main()
