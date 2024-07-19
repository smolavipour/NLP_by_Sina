import streamlit as st
import sys
import json
sys.path.append('./')

from Input_handler import input_handler_model
from Input_handler.input_handler_model import plot_chart, parse_names


from langchain_core.messages import HumanMessage, AIMessage

#import dotenv
import os

#dotenv.load_dotenv()

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: purple;'>StockmarketBot</h1>", unsafe_allow_html=True)
conversation_mode = True


@st.cache_resource
def load_models():
    input_handler_model_obj = input_handler_model.input_handler()
    return input_handler_model_obj



input_handler_model_obj = load_models()


if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
if "chart_history" not in st.session_state:
    st.session_state.chart_history=[]

user_query = None




c1,c2 = st.columns([6,1],gap="large")
with c1:
    conversation_box = st.container(height=1000)
    if conversation_mode:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                conversation_box.chat_message("Human").write(message.content)
            elif isinstance(message, AIMessage):
                conversation_box.chat_message("AI").write(message.content)
            elif isinstance(message, tuple):
                fig_chart,_ = message
                with conversation_box:
                    c_1,c_2,c_3 = st.columns([1,1,1],gap="large")
                    with c_2:
                        st.pyplot(fig_chart)
    
    chat = st.chat_input("Your message")
    user_query = chat
            
    if user_query is not None and user_query!="":        
        conversation_box.chat_message("Human").write(user_query)
        conversation_history = "\n".join([a.pretty_repr() for a in st.session_state.chat_history if not isinstance(a,tuple)])
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        response = input_handler_model_obj.data_ingestion({"question":user_query})   
        
        input_handler_response = "The query is created."
        conversation_box.chat_message("AI").write(input_handler_response)
        st.session_state.chat_history.append(AIMessage(input_handler_response))
        
        try:
            fig_chart, ax_chart, _ = plot_chart(response)
            st.session_state.chat_history.append((fig_chart, ax_chart))
            try:
                with conversation_box:
                    c_1,c_2,c_3 = st.columns([1,1,1],gap="large")
                    with c_2:
                        st.pyplot(fig_chart)
            except Exception as e:
                print(e)
                pass
        except Exception as e:
            print(f"An error occurred: {e}")
        
with c2:
    if st.button("Clean chat 🧹"):
        st.session_state.chat_history=[]



        
    

