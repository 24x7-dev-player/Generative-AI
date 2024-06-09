llm_config = {"model": "gpt-4-turbo"}
from autogen.coding import LocalCommandLineCodeExecutor
executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="/content/current",
)
from autogen import ConversableAgent, AssistantAgent
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)
code_writer_agent_system_message = code_writer_agent.system_message
print(code_writer_agent_system_message)



import datetime
today = datetime.datetime.now().date()
message = f"Today is {today}. "\
"Create a plot showing stock gain YTD for NVDA and TLSA. "\
"Make sure the code is in markdown code block and save the figure"\
" to a file ytd_stock_gains.png."""
chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message,
)
import os
from IPython.display import Image
Image()


def get_stock_prices(stock_symbols, start_date, end_date):
    """Get the stock prices for the given stock symbols between
    the start and end dates.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: The stock prices for the given stock
        symbols indexed by date, with one column per stock
        symbol.
    """
    import yfinance

    stock_data = yfinance.download(
        stock_symbols, start=start_date, end=end_date
    )
    return stock_data.get("Close")
  
  
def plot_stock_prices(stock_prices, filename):
    """Plot the stock prices for the given stock symbols.

    Args:
        stock_prices (pandas.DataFrame): The stock prices for the
        given stock symbols.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index, stock_prices[column], label=column
                )
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig(filename)
    
    
    executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="/content/current",
    functions=[get_stock_prices, plot_stock_prices],
)
    
code_writer_agent_system_message += executor.format_functions_for_prompt()
print(code_writer_agent_system_message)