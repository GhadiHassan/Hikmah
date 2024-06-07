
#this is to fix sqlite chroma error from streamlit + add pysqlite3-binary to requirements.txt
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import time
#from dotenv import load_dotenv  # Load environment variables from a file
import base64 #for locally stored images
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)



embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

prompt_template = """You are Aljazari, please use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
2. If you find the answer, write the answer in a concise way with five sentences maximum.
3. don't refer to the provided context in your answer.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


@st.cache_data #(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file,header):
    bin_str = get_base64_of_bin_file(png_file)
    header_str = get_base64_of_bin_file(header)
    page_bg_img = '''
    <style>
    body {
        
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    # the header class .st-emotion-cache-18ni7ap
    # the middle .st-emotion-cache-1r4qj8v
    header = '''
    <style>
    .st-emotion-cache-12fmjuu{
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    } 
    </style>
    ''' % header_str
    test = '''
    <style>
    .st-emotion-cache-bm2z3a{
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str

    footer = '''
    <style>
    .st-emotion-cache-uhkwx6{
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % header_str
    

    st.markdown(test, unsafe_allow_html=True)
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(footer, unsafe_allow_html=True)

    return

def set_text_color():
    text = '''
    <style>
    h1{
        color: rgb(254 202 111);
    }
    </style>
    '''
    st.markdown(text, unsafe_allow_html=True)
    return

def getBotResponse(query):
    '''
    This function will take the user query, and returns the bot response

    Parameters: 
        prompt (string): user query

    Return:
        response: bot response
    '''
    print(query)
    result = retrievalQA.invoke({"query": query})
    return  result["result"]


def main():

    set_png_as_page_bg('background1.png','background1.png')
    set_text_color()

    #set the color of conversation text
    st.markdown(
        """
        <style>
        .st-emotion-cache-1flajlm{
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the title
    st.title("Ask Al-Jazari")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            


    # Accept user input
    if prompt := st.chat_input("Enter your question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = getBotResponse(prompt)
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})




if __name__ == '__main__':
    main()