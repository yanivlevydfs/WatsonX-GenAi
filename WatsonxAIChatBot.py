import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
import json
import os
import pandas as pd

st.title('Watsonx AI Chatbot 🤖')
st.caption("🚀 A chatbot powered by watsonx.ai - Yaniv Levy IBM")

with st.sidebar:
    watsonx_api_key = st.text_input("Watsonx API Key", key="watsonx_api_key", value=os.environ.get("API_KEY"), type="password")
    data = [['Frankfurt', "https://eu-de.ml.cloud.ibm.com"], ['Dallas', "https://us-south.ml.cloud.ibm.com"], ['London', "https://eu-gb.ml.cloud.ibm.com"], ['Tokyo', "https://jp-tok.ml.cloud.ibm.com"]]
    df = pd.DataFrame(data, columns=['Name', 'Location'])
    hostValues = df['Name'].tolist()
    hostOptions = df['Location'].tolist()
    dic = dict(zip(hostOptions, hostValues))
    watsonx_url = st.sidebar.selectbox('Please choose your server?', hostOptions, format_func=lambda x: dic[x],index=1,placeholder="Select watsonx url...")
    st.write('You selected:',watsonx_url)    
    watsonx_model = st.selectbox('Please choose your LLM?',
    ('bigcode/starcoder', 'bigscience/mt0-xxl', 'eleutherai/gpt-neox-20b', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2', 'ibm/granite-13b-chat-v1', 'ibm/granite-13b-chat-v2', 'ibm/granite-13b-instruct-v1', 'ibm/granite-13b-instruct-v2', 'ibm/mpt-7b-instruct2', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-2-70b-chat'),index=1,placeholder="Select watsonx model...")
    st.write('You selected:', watsonx_model)
    decoding_method = st.text_input('Decoding Method:', 'sample')
    st.write('You selected:', decoding_method)
    max_new_tokens = st.text_input('Max New Tokens:', '200')
    st.write('You selected:', max_new_tokens)
    temperature = st.text_input('Temperature:',0.5)
    st.write('You selected:', temperature)
    watsonx_model_params = json.dumps({'Decoding Method': decoding_method, 'Max New Tokens':int(max_new_tokens),'Temperature': float(temperature)})
    st.write(watsonx_model_params);
    
if not watsonx_api_key:
    st.info("Please add your watsonx API key to continue.")
else :
    st.info("setting up to use: " + watsonx_model)
    my_credentials = { 
        "url"    : watsonx_url, 
        "apikey" : watsonx_api_key
    }
    params = json.loads(watsonx_model_params)
    project_id  = os.environ.get("PRJ_ID")
    space_id    = None
    verify      = False
    model = Model( watsonx_model, my_credentials, params, project_id, space_id, verify )   
    if model :
        st.info("done")
 
if 'messages' not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}] 

for message in st.session_state.messages: 
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    if model :
        response = model.generate_text(prompt)
    else :
        response = "You said: " + prompt
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})