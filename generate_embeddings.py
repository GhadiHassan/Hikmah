from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


loader = PyPDFLoader("documents\AlhiyalBook.pdf") #set the path to the data file
docs_before_split = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

print(docs_after_split[0])

#set the embedding model fro sentence transformers
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#embedd data and store in db
vectorstore = Chroma.from_documents(docs_after_split, embedding_function,persist_directory="./chroma_db")
print(vectorstore._collection.count())



