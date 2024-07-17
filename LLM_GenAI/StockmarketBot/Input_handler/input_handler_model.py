import dotenv
import os
import re
import json
from datetime import datetime

from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools

import yfinance as yf
import matplotlib.pyplot as plt


os.chdir('C:/Users/esinmol/Documents/sina/NLP_by_Sina/LLM_GenAI/StockmarketBot/Input_handler')
dotenv.load_dotenv()

def parse_names(message: str) -> list[dict]:
    # Define the regular expression pattern to match JSON blocks
    pattern = r'<OUTPUT>([^<]+)</OUTPUT>'

    # Find the content within <NAMES> tags
    match = re.search(pattern, message)

    if match:
        # Extract the matched content
        names_string = match.group(1)
        # Split the content by commas to get individual company names
        company_names = [name.strip() for name in names_string.split(',')]
        return company_names
    else:
        print("No match found")
        return []

class input_handler:
    def __init__(self):
        '''
        The endpoint can be either "HuggingFace" or "Turing"
        '''
        
        API_KEY = os.environ.get("HuggingFace_API_KEY")
        #repo_id = "mistralai/Mistral-7B-v0.3"
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        #repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        #repo_id = "meta-llama/Llama-2-7b-hf"

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=API_KEY
        )
        
        ############################
        ##   Agent company names  ##
        ############################
        message_template_company_names = """ You are a stock exchange expert and would answer very accurate and shortly.
        You will recieve a question and you need to first extract the company names and their corresponding tickers.

        # Instruction
        1) Based on user question extract the company names that is mentioned. 
        2) Then for every company name identify the corresponding ticker name (i.e., stock symbol)
        3) List tickers in the following format: <OUTPUT>ticker_1,ticker_2,...,ticker_n</OUTPUT>.
        
        # Rules
        - Only print the list and nothing else.
        - Do not assume when you are not confident
        - **Answer only** the OUTPUT block <OUTPUT>ticker_1,ticker_2,...,ticker_n</OUTPUT>.

        # User Question
        User query is:
        <question>
        {question}
        </question>
        """

        self.prompt_company_names = PromptTemplate(
            template=message_template_company_names,
            input_variables=["question"]
        )

        self.chain_company_names = self.prompt_company_names | self.llm

def plot_chart(ticker_names):
    price = {}
    fig, ax = plt.subplots(figsize=(6, 6))
    for ticker in ticker_names:
        # Get the data
        data = yf.download(ticker,'2016-01-01','2024-06-01')

        # Plot the close price
        price[ticker] = data['Adj Close']

        ax.plot(price[ticker].index, price[ticker].values, label=ticker)
        



    ax.legend()
    ax.set_title('Stock historical data')
    ax.set_xlabel('time')
    ax.set_ylabel('price (USD)')

    return fig, ax, price