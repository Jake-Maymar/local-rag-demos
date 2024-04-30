#!/bin/env python3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            message(msg, is_user=is_user, key=str(i), avatar_style="avataaars", seed=123)

            
            
        else:
            message(msg, is_user=is_user, key=str(i), avatar_style="bottts", seed="Precious")
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            if st.session_state["assistant"].chain:
                agent_text = st.session_state["assistant"].ask(user_text)
            else:
                agent_text = "Please upload a document first."

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["file_uploaded"] = False

    if st.session_state["file_uploader"]:
        for file in st.session_state["file_uploader"]:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                st.session_state["assistant"].ingest(file_path)
            os.remove(file_path)

        st.session_state["file_uploaded"] = True
    else:
        st.session_state["assistant"].remove_document()
        st.warning("Please upload a document.")


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        st.session_state["file_uploaded"] = False

    st.header("Chat with your PDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()