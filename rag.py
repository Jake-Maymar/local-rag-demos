from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3:latest")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use fifteen sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        print(f"Ingesting PDF file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        print(f"Number of loaded documents: {len(docs)}")
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        print(f"Number of chunks after splitting: {len(chunks)}")

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        print("Vector store created")
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.2,
            },
        )
        print("Retriever set up")

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        print("Chain set up")

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        print(f"Query: {query}")
        retrieved_docs = self.retriever.get_relevant_documents(query)
        print(f"Number of retrieved documents: {len(retrieved_docs)}")
        for doc in retrieved_docs:
            print(f"Document: {doc.page_content}")

        return self.chain.invoke(query)

    def remove_document(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None