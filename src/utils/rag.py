    """RAG script to handle text retreival and generation.
   
    """
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
            model=model_name, temperature=temperature, repeat_penalty=repeat_penalty, timeout=120)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=200)

        self.use_history = use_history

        if self.use_history:
            # contextualize q prompt
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(
                        "Question:```{question}```")
                ]
            )
            self.contextualize_q_chain = contextualize_q_prompt | self.model | StrOutputParser()
            # Define the system message template

            system_template = """"You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

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

            self.docs = TextLoader(
                file_path=file_path,
                autodetect_encoding=True

            ).load()
            chunks = self.text_splitter.split_documents(self.docs)
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

    def _contextualized_question(self, input: dict):
        if input.get("chat_history"):
            return self.contextualize_q_chain
        else:
            return input["question"]

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chain(self):

        if self.use_history:
            # self.chain = ConversationalRetrievalChain.from_llm(
            #     self.model, self.retriever,
            #     return_source_documents=True,
            #     verbose=False,
            #     combine_docs_chain_kwargs={"prompt": self.prompt})
            self.chain = (
                RunnablePassthrough.assign(
                    context=self._contextualized_question | self.retriever | self._format_docs
                )
                | self.prompt
                | self.model
                | StrOutputParser()
            )

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
