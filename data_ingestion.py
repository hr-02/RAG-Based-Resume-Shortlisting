import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import OllamaEmbeddings


DATA_PATH = "./data/synthetic-resume.csv"
FAISS_PATH = "./vectorstore/faiss"
EMBEDDING_MODEL = "llama3"


def initiate_vectorstore():
  df = pd.read_csv(DATA_PATH)
  loader = DataFrameLoader(df, page_content_column="Resume")
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40,
  )
  embedding_model = OllamaEmbeddings(model="llama3")
  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)
  vectorstore_db = FAISS.from_documents(document_chunks, embedding_model)

  vectorstore_db.save_local(FAISS_PATH)


if __name__ == "__main__":
  initiate_vectorstore()