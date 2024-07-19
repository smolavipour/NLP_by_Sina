#import dotenv
import os
import re
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools

import yfinance as yf
import matplotlib.pyplot as plt


#os.chdir('C:/Users/esinmol/Documents/sina/NLP_by_Sina/LLM_GenAI/StockmarketBot/Input_handler')
#dotenv.load_dotenv()

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
    
def parse_date(message):
    # Define the regular expression pattern to match JSON blocks
    pattern = r"<OUTPUT>(.*?)<\/OUTPUT>"
    
    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, message, re.DOTALL)
    try:
        json_text = matches[0]
    except:
        print('Problem with json conversion.')
        return None
    json_text = json_text.replace("'", '"').strip()
    
    try:
        schema = json.loads(json_text)
    except:
        print('Problem with json conversion.')
        return None

    return schema


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

        ###################
        ##   Agent Date  ##
        ###################
        message_template_date = """ You are supposed to pick-up dates and date periods from user's input.
        User input is usually a query to get financial historical data. Your task is to pick up the date or period of interest.
        For calculating relative dates, note that today is **{today}**.
        Provide the final answer in a JSON following the schema provided in <schema> block.
        The JSON output is suppused to be placed <OUTPUT>JSON</OUTPUT> block.

        # Rules
        - Do not assume when you are not confident
        - **Answer only** the OUTPUT block <OUTPUT>JSON</OUTPUT>.
        - If you needed to compute relative dates (last year, last month), today is **{today}**.
        - If user asks for previous quarters, first identify what quarter is now. Then compute relative quarter and months accordingly.
        
        # Schema
        To express a date period, we determin a lower bound "gt", and an upper bound "lt". We use the format %Y-%M-%D.
        To compute relative dates, assume
        <schema>
        {{
         "gt" : "%Y-%M-%D",
         "lt" : "%Y-%M-%D"
        }}
        </schema>

        # Examples
        ## Example 1
        User: July 2023
        AI: <OUTPUT>
        {{
         "gt" : "2023-07-01",
         "lt" : "2023-07-31"
        }}
        </OUTPUT>

        ## Example 2
        User: Last month
        AI: <OUTPUT>
        {{
         "gt" : "{last_month}",
         "lt" : "{today}"
        }}
        </OUTPUT>        

        # User Question
        User query is:
        <question>
        {question}
        </question>
        """
        today = datetime.today()
        self.prompt_date = PromptTemplate(
            template=message_template_date,
            input_variables=["question"],
            partial_variables={"today": today.strftime("%Y-%m-%d"),
                               "last_month":(today + relativedelta(months=-1)).strftime("%Y-%m-%d")}
        )
        self.chain_date = self.prompt_date | self.llm | parse_date
    
    def data_ingestion(self, message):
        res_json = {}

        ##############
        # Tickers        
        try:
            response = self.chain_company_names.invoke(message)
            ticker_names = parse_names(response)
        except:
            print('LLM problem fetching ticker names')
            ticker_names = []
        res_json.update({'ticker':ticker_names})
        
        # Date        
        try:
            range = self.chain_date.invoke({"question":message})
        except:
            print('LLM problem fetching date')
            today = datetime.today()
            range = {"gt" : today + relativedelta(months=-36),
                    "lt" : today}
            
        res_json.update({'range':range})

        return res_json

def plot_chart(output_json):
    ticker_names = output_json['ticker']
    range_lt = output_json['range']['lt']
    range_gt = output_json['range']['gt']

    price = {}
    fig, ax = plt.subplots(figsize=(6, 6))
    for ticker in ticker_names:
        # Get the data
        data = yf.download(ticker,range_gt,range_lt)

        # Plot the close price
        price[ticker] = data['Adj Close']

        ax.plot(price[ticker].index, price[ticker].values, label=ticker)
        
    ax.legend()
    ax.set_title('Stock historical data')
    ax.set_xlabel('time')
    ax.set_ylabel('price (USD)')

    return fig, ax, price
