from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

os.environ["HUGGINGFACEHUB_API_TOKEN"]= "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"

loader = PyPDFLoader("documents\AlhiyalBook.pdf")
docs_before_split = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

print(docs_after_split[0])

# huggingface_embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
#     model_kwargs={'device':'cpu'}, 
#     encode_kwargs={'normalize_embeddings': True}
# )
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#sample_embedding = np.array(embedding_function.embed_query(docs_after_split[0].page_content))
# #print("Sample embedding of a document chunk: ", sample_embedding)
# print("Size of the embedding: ", sample_embedding.shape)


vectorstore = Chroma.from_documents(docs_after_split, embedding_function,persist_directory="./chroma_db")
print(vectorstore._collection.count())
# # Use similarity searching algorithm and return 3 most relevant documents.
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

# system = "You are a helpful assistant."
# human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# chain = prompt | chat
# print(chain.invoke({"text": "Explain the importance of low latency LLMs."}))



# prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
# 1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
# 2. If you find the answer, write the answer in a concise way with five sentences maximum.

# {context}

# Question: {question}

# Helpful Answer:
# """

# PROMPT = PromptTemplate(
#  template=prompt_template, input_variables=["context", "question"]
# )

# retrievalQA = RetrievalQA.from_chain_type(
#     llm=chat,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )

# # Call the QA chain with our query.
# query = "who is aljazari"
# result = retrievalQA.invoke({"query": query})
# print(result['result'])

# # text_splitter = CharacterTextSplitter(chunk_size=280, chunk_overlap=0)
# # docs = text_splitter.split_documents(document)
# # print(len(docs))
# # embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# # query_result = embedder.embed_query("hi how are you")
# # print(len(query_result))


# # db2 = Chroma.from_documents(docs, embedder, persist_directory="./chroma_db")
# # print(db2)



