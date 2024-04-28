from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import pathlib
import pickle
import os


class RAG:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model_name: str = "phi3", temperature: float = 0.8, repeat_penalty=1.5, use_history: bool = False) -> None:
        self.model = ChatOllama(
            model=model_name, temperature=temperature, repeat_penalty=repeat_penalty)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=200)

        self.use_history = use_history

        if self.use_history:

            # Define the system message template

            system_template = """Answer the following question based only on the provided context in 5 sentences,
            if you cannot find the answer from the pieces of context, just say that you don't know, don't try to 
            make up an answer.
            ----------------
            {context}"""

            # Create the chat prompt templates
            messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    "Question:```{question}```")
            ]
            self.prompt = ChatPromptTemplate.from_messages(messages)

        else:
            self.prompt = PromptTemplate.from_template(

                """<|system|>\nYou are assistant for question-answering tasks. Use the following context to answer the question.
             If you don't know the answer, just say you don't know and do not make up any answer. Use three sentences
             and be concise in your answer.<|end|>
             <|user|>
             {question}<|end|>
             <|assistant|>""")

    def ingest(self, file_path: str):
        if pathlib.Path("../data/chunks/chunks.pkl").exists():
            chunks = pickle.load(open("../data/chunks/chunks.pkl", "rb"))

        else:

            docs = TextLoader(
                file_path=file_path,
                autodetect_encoding=True

            ).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            pickle.dump(chunks, open("../data/chunks/chunks.pkl", "wb"))

        db = Chroma

        if len(os.listdir("../db")) > 0:

            vector_store = db(persist_directory="../db",
                              embedding_function=FastEmbedEmbeddings(cache_dir="../cache"))

        else:
            vector_store = db.from_documents(
                documents=chunks, persist_directory="../db",
                embedding=FastEmbedEmbeddings(cache_dir="../cache"))

            vector_store.persist()

        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5,
            },
        )

    def get_chain(self):

        if self.use_history:
            self.chain = ConversationalRetrievalChain.from_llm(
                self.model, self.retriever,
                return_source_documents=True,
                verbose=False,
                combine_docs_chain_kwargs={"prompt": self.prompt})

        else:
            self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                          | self.prompt
                          | self.model
                          | StrOutputParser())

        return self.chain

    def ask(self, query: str):
        if not self.chain:
            return "Documents didnt loaded!!!"

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
