from dataclasses import dataclass
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from utils.rag import RAG
import sys
from loguru import logger

log = logger.info


st.markdown("<h1 style='text-align: center; color: blue;'>GeetaVani<br>🛕🐚🔱ૐ धर्मो रक्षति रक्षितः</h1>",
            unsafe_allow_html=True)


USER = HumanMessage
ASSISTANT = AIMessage
MESSAGES = "messages"


def ingest_data(_rag: RAG):
    with st.spinner("Loading..."):
        _rag.ingest("../data/geeta_txt/geeta.txt")
    return _rag


def initialize_session_state(model_name: str = "phi3",
                             temperature: float = 0.8,
                             use_history: bool = False):
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES]: list = [
            AIMessage(content="Hi, how can I help you ?")]
        # AIMessage(content="Hi! How can I help you?")]
    if "llm_chain" not in st.session_state:
        rag = RAG(model_name, temperature, use_history=use_history)
        rag = ingest_data(rag)
        st.session_state["llm_chain"] = rag.get_chain()


def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


def main(use_history=False):

    msg: AIMessage
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.type).write(msg.content)

    st.session_state[MESSAGES].clear()

    prompt: str = st.chat_input("Ask anything...")

    if prompt:
        human_msg = HumanMessage(content=prompt)
        st.chat_message(human_msg.type).write(human_msg.content)

        with st.spinner("Please wait.."):
            llm_chain = get_llm_chain_from_session()

            if use_history:

                # st.write(st.session_state[MESSAGES])

                result: dict = llm_chain.invoke({"question": prompt,
                                                 "chat_history": st.session_state[MESSAGES]})

                response = result.get("answer", "")

            else:
                response: str = llm_chain.invoke(prompt)

            ai_msg = AIMessage(content=response)

            st.session_state[MESSAGES].extend([human_msg, ai_msg])

            st.chat_message(ai_msg.type).write(response)


if __name__ == "__main__":
    use_history = False
    model_name = "phi3"
    if len(sys.argv) > 1:
        if sys.argv[1] not in ["llama3", "phi3"]:
            raise Exception("Model not yet supported")
        model_name: str = sys.argv[1]

        if len(sys.argv) == 3:
            use_history: bool = (
                True if sys.argv[2] == "-use_history" else False)

    log(f"Model : {model_name}, Use history: {use_history}")
    initialize_session_state(
        model_name=model_name, temperature=0, use_history=use_history)
    main(use_history=use_history)
