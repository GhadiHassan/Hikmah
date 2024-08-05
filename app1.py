from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import gradio as gr
import streamlit as st

embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


chat = ChatGroq(temperature=0, groq_api_key=st.secrets["groq_api_key"], model_name="mixtral-8x7b-32768")


prompt_template = """You are Aljazari, please use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
2. If you find the answer, write the answer in a concise way with five sentences maximum.

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

# Call the QA chain with our query.
# query = "who is aljazari"
# result = retrievalQA.invoke({"query": query})
# print(result['result'])


def echo(message, history):
    query = message
    result = retrievalQA.invoke({"query": query})
    #print(result)
    return result["result"]

demo = gr.ChatInterface(fn=echo, title="Aljazari Bot")
demo.launch(share=True)


