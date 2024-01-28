![enter image description here](https://media.licdn.com/dms/image/D4D12AQHHPc7TKWSKLQ/article-cover_image-shrink_720_1280/0/1695548748450?e=1712188800&v=beta&t=bpwZgrO9UmbJ6EbXyMkUicAn9jKofrRn_bbOasBiZkE) 

# Calling an LLM Model on IBM Watsonx.ai Service Using IBM Cloud Code Engine
Yaniv Levy - IBM 
https://www.linkedin.com/in/yanivlevy
February 2014

Language models have become increasingly popular for natural language understanding and generation tasks. IBM  [Watsonx.ai](http://watsonx.ai/)  is a cloud-based service that provides access to powerful language models, such as Large Language Models (LLMs).  [Watsonx.ai](http://watsonx.ai/)  is part of the IBM watsonx platform that brings together new generative AI capabilities, powered by foundation models and traditional machine learning into a powerful environment spanning the AI lifecycle. With  [watsonx.ai](http://watsonx.ai/), you can train, validate, tune and deploy generative AI, foundation models and machine learning capabilities with ease and build AI applications in a fraction of the time with a fraction of the data. In this article, we'll explain a Python code that interacts with an LLM hosted on IBM  [Watsonx.ai](http://watsonx.ai/)  using  [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about). This code creates a simple chatbot interface for interacting with the model.

![alt text](https://github.ibm.com/Yaniv-Levy/WatsonX-GenAi/blob/main/TestGranite-ezgif.com-video-to-gif-converter.gif?raw=true)

## Prerequisites for deploying on Cloud

Before diving into the code, ensure you have the following prerequisites in place:

1.  **IBM Cloud Account:**  You'll need an IBM Cloud account to access the  [Watsonx.ai](http://watsonx.ai/)  service and  [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about).
2.  **Project ID:**  Within your IBM Cloud account you will deploy the Watsonx service, create a project and associate it with a Watson Machine Learning Service.
3.  **API Key:**  To use the  [foundation models Python library](https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html), you need an IBM Cloud API key. You can create one in your IBM Cloud account. To create an API key for your user identity In the IBM Cloud console, go to  **Manage**  >  **Access (IAM)**  >  **API keys**. and Click  **Create an IBM Cloud API key**. For security reasons, the API key is only available to be copied or downloaded at the time of creation. If the API key is lost, you must create a new API key.
4.  **IBM Cloud Code Engine:**  Set up an  [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about)  environment for running this code.

## Prerequisites for running on localhost

 1. Install PHP on your local machine
 2. Import the following modules from the code below via PIP
 3. Download the repository from GitHub - (https://github.ibm.com/Yaniv-Levy/WatsonX-GenAi)
 4. Hardcode in your PHP class two environment variables - PRJ_KEY & API_KEY ( instructions continue below in how to get them)
 5. on CMD or any Shell execute the following: 
 

    streamlit run WatsonxAIChatBot.py 

## Understanding the Python Code

A simple chatbot application created using the Streamlit library.

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

Let's break down the Python code step by step:

1.  **Importing Dependencies:**  The code begins by importing necessary libraries, including Streamlit, for building the chatbot interface, and the ibm_watson_machine_learning library for interacting with the  [Watsonx.ai](http://watsonx.ai/)  service.

2.  **Streamlit Setup:**  The Streamlit application is created with a title and a caption to provide some information about the chatbot.

3.  **Sidebar Inputs:**  The code creates a sidebar with input fields for the Watsonx API key, Watsonx URL, model name, and model parameters. The API key can be entered manually, or you can provide it as an environment variable. The default Watsonx URL is set to "[https://us-south.ml.cloud.ibm.com](https://us-south.ml.cloud.ibm.com/)," and the model name and parameters are also provided.

4.  **Credentials and Model Initialization:**  If the API key is provided, the code sets up the Watsonx credentials and initializes the model using the provided API key, URL, model name, and parameters. It also checks for project and space IDs.

5.  **Message History:**  A session state variable, messages, is used to store the chatbot's conversation history. If this variable doesn't exist, it is initialized with a default message from the chatbot.

6.  **User Interaction:**  The code handles user interaction by allowing users to input prompts. When a user enters a prompt, it is added to the chat history, and the chatbot generates a response using the initialized model.

7.  **Generating Responses:**  If the model is successfully initialized, the user's prompt is used to generate a response from the model. If the model initialization fails, a simple echo response is generated.

8.  **Displaying Responses:**  The user's input and the chatbot's response are displayed in the chat interface.

## Running the Code on IBM Cloud Code Engine

Deploying an application from a Git repository on  [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about)  is a straightforward process that leverages the platform's integration with Git-based source control systems. To get started, you first need to create a project on IBM Cloud Code Engine. Within the project you will be able to create an application and connect it to your Git repository. Once connected, you can set up deployment triggers to automatically build and deploy your application whenever changes are pushed to the specified branch of your Git repository. This integration streamlines the deployment process, ensuring that your application is always up-to-date with the latest code changes.

In order to deploy our application we will choose a Dockerfile based build. A Docker build creates a container based on how you describe it in a Dockerfile. The Dockerfile is then committed along with your source code to create the container.

While you can use either strategy for your build, you might choose Dockerfile, if, for example,

-   Your programming environment is not supported by Buildpacks.
-   Your project build must install additional packages in the container.

# app/Dockerfile

    FROM python:3.9-slim
    
    WORKDIR /app
    
    COPY requirements.txt ./
    COPY *.py ./
    
    RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        software-properties-common \
        git \
        && rm -rf /var/lib/apt/lists/*
    
    RUN pip3 install -r requirements.txt
    
    EXPOSE 8501
    
    HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
    
    ENTRYPOINT ["streamlit", "run", "WatsonxAIChatBot.py", "--server.port=8501", "--server.address=0.0.0.0"]

This Dockerfile is used to build a Docker image for running a Python application using Streamlit. Streamlit is a popular Python library for creating web applications with minimal code, often used for data visualization and interactive web interfaces. Let's break down each section of the Dockerfile:

1. FROM python:3.9-slim:

- This line specifies the base Docker image to use. In this case, it's based on Python 3.9-slim, which is a lightweight version of Python 3.9. This image is commonly used for Python applications to minimize image size.

2. WORKDIR /app:

- This line sets the working directory inside the Docker container to /app. All subsequent commands will be executed in this directory.

3. COPY requirements.txt ./:

- This line copies the requirements.txt file from the host machine (the directory where the Dockerfile is located) to the /app directory in the container. This file typically lists the Python packages and their versions required by the application. in our case the requirements file will contain two lines referencing:  _streamlit_  and  _ibm_watson_machine_learning._

4. COPY *.py ./:

- This line copies all Python files (with a .py extension) from the host machine to the /app directory in the container. These are the application source code files.

5. RUN apt-get update && apt-get install -y ...:

- These lines run Linux package manager commands (`apt-get`) to update the package repository and install several system-level dependencies required for building and running the application. These dependencies include build-essential, curl, software-properties-common, and git. After installation, the package cache is cleaned up (`rm -rf /var/lib/apt/lists/*`) to reduce the image size.

6. RUN pip3 install -r requirements.txt:

- This line uses pip3 to install the Python packages listed in requirements.txt. It installs the necessary Python libraries for the Streamlit application to run.

7. EXPOSE 8501:

- This line specifies that port 8501 should be exposed to the network. This is the default port that Streamlit applications use.

8. HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health:

- This line defines a health check command for Docker. It uses curl to check the health of the application by attempting to access a specific health endpoint. If the health check fails, the container is considered unhealthy.

9. ENTRYPOINT ["streamlit", "run", "WatsonxAIChatBot.py", "--server.port=8501", "--server.address=0.0.0.0"]:

- This line specifies the command that will be executed when the Docker container starts. It runs the Streamlit application named WatsonxAIChatBot.py and configures it to listen on port 8501 and accept requests from any network address (`0.0.0.0`). This command is what starts the Streamlit application when the Docker container is launched.

In summary, this Dockerfile sets up an environment for running a Streamlit-based Python application, installs required system dependencies and Python packages, exposes the application on port 8501, and defines a health check for the Docker container. When you build and run a container from this image, it will execute the specified Streamlit application.

![](https://media.licdn.com/dms/image/D4D12AQEjCwUdo78WTw/article-inline_image-shrink_1000_1488/0/1695546679317?e=1712188800&v=beta&t=4eCT9RRVl5ilPb0rrQvTmIoLxm3WwsAcXKiRpbFp5Oc)

Exposing secrets to the application as environment variables

Additionally, you can configure environment variables, specify container configurations, and scale your application as needed, all from the IBM Cloud Code Engine interface. This enables developers to easily manage and deploy their applications with confidence, knowing that their code is securely hosted on the cloud and can be seamlessly updated through their Git repository.

When deploying an application from a Git repository on IBM Cloud Code Engine, it's essential to emphasize the value of storing sensitive information, such as API keys and project IDs, as secrets within the Code Engine project. These secrets can be securely shared with the application by having the platform inject them as environment variables in the execution context. The added value lies in the security and access control of IBM Cloud's secrets management.

![](https://media.licdn.com/dms/image/D4D12AQEvAf9ODw6fqw/article-inline_image-shrink_1500_2232/0/1695546996684?e=1712188800&v=beta&t=DkKXcQGLQmcNmIxeZhnwZ-aynY7A1IwqjDKoOEKVCak)

Code Engine project secrets

By storing secrets as project-specific environment variables, you ensure that sensitive information remains protected. IBM Cloud's secrets management is designed to be highly secure, with access restricted to the user who has created them. This means that even within a shared development or deployment environment, only authorized users can access these secrets. This robust security mechanism helps safeguard sensitive credentials, ensuring they are not exposed accidentally or to unauthorized personnel.

In summary, using IBM Cloud Code Engine's secrets management to store and share API keys and project IDs as environment variables provides a robust security layer, ensuring that sensitive information remains confidential and accessible only to those who need it, enhancing the overall security posture of your application.

![](https://media.licdn.com/dms/image/D4D12AQHyCE6em-UZ9w/article-inline_image-shrink_1000_1488/0/1695547342819?e=1712188800&v=beta&t=cj6F80FE3sd-M3DVy9QIrELTRhhuaaMXEh-CVXYPoMA)

Once built the container image for your application will be pushed to IBM Cloud Container Registry. IBM Cloud Container Registry service offers a robust and secure solution for managing Docker images. One of its standout features is the built-in Vulnerability Advisor, which enhances image security and compliance. When Docker images are pushed to the Container Registry, Vulnerability Advisor automatically scans these images for potential security issues and vulnerabilities. It meticulously checks for vulnerable packages within specific Docker base images and scrutinizes known vulnerabilities in application configuration settings. In the event that vulnerabilities are detected, the service provides detailed information about these security risks. This valuable feedback empowers users to take proactive measures to resolve security issues promptly, ensuring that containers are not deployed from vulnerable images. Container Registry also offers scalability and high availability, allowing users to set up their own image namespace within a multi-tenant, encrypted private registry, hosted and managed by IBM. Users can securely store and share private Docker images with others in their IBM Cloud account. Additionally, Container Registry provides quota limits for storage and pull traffic, including free storage and pull traffic up to a specified limit. Users can set custom quota limits to manage resource consumption effectively, aligning with their preferred payment level. These combined features make IBM Cloud Container Registry a dependable solution for container image management, security, and compliance.

## Enjoy your conversation with the LLM

To run this code on IBM Cloud Code Engine:

1.  **Deploy the Application:**  Deploy the code as a Streamlit application on IBM Cloud Code Engine. Make sure you have the required environment variables set, including the Watsonx API key and project ID.

2.  **Access the Application:**  Once deployed, you can access the chatbot interface through the provided URL.

3.  **Interact with the Chatbot:**  You can now interact with the chatbot by entering prompts in the input field. The chatbot will respond based on the initialized LLM model.

![](https://media.licdn.com/dms/image/D4D12AQHBoZ0li2d92Q/article-inline_image-shrink_1000_1488/0/1695653916831?e=1712188800&v=beta&t=VvMxh75WZlk6b-0JHfh6_3tS1BUTuHfbn2eUAfgh23I)

**

## This screenshot is quite old, i was too lazy to amend it, meanwhile i have improved the code with several new features. guess what?

**

## Conclusion

In this article, we've explained how to use Python code to create a chatbot interface that interacts with an LLM model hosted on IBM  [Watsonx.ai](http://watsonx.ai/)  through  [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about). By following the provided steps, you can set up and run this code to build your own chatbot powered by  [Watsonx.ai](http://watsonx.ai/)'s language models. This can be a valuable tool for various natural language understanding and generation tasks.

## Happy Coding - Yaniv



