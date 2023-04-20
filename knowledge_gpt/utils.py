import re
from io import BytesIO
from typing import Any, Dict, List

import docx2txt
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError
from pypdf import PdfReader

from knowledge_gpt.embeddings import OpenAIEmbeddings
from knowledge_gpt.prompts import STUFF_PROMPT

import faiss
import numpy as np


@st.experimental_memo()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.experimental_memo()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


@st.experimental_memo()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache(allow_output_mutation=True)
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache(allow_output_mutation=True, show_spinner=False)
def embed_docs(docs: List[Document]) -> FAISS:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.get("OPENAI_API_KEY")
        )  # type: ignore

        doc_texts = [doc.page_content for doc in docs]
        chunked_embeddings = embeddings.embed_documents(doc_texts)
        flattened_embeddings = [
            embedding for doc_embeddings in chunked_embeddings for embedding in doc_embeddings
        ]
        #print(flattened_embeddings[:10])

        print(np.shape(flattened_embeddings))

        # Create the FAISS index
        index_ = faiss.IndexFlatL2(int(np.shape(flattened_embeddings[0])[0]))
        index_.add(np.array(flattened_embeddings, dtype=np.float32))
        index = FAISS(embeddings.embed_documents, index_, docstore=docs, index_to_docstore_id=list(range(len(docs))))
        print(index)

        # index = FAISS.from_documents(docs, embeddings)

        return index

# @st.cache(allow_output_mutation=True)
# def embed_docs(docs: List[Document]) -> VectorStore:
#     """Embeds a list of Documents and returns a VectorStore object"""
    
#     if not st.session_state.get("OPENAI_API_KEY"):
#         raise AuthenticationError(
#             "Enter your OpenAI API key in the sidebar. You can get a key at"
#             " https://platform.openai.com/account/api-keys."
#         )
#     else:
#         # Embed the chunks
#         embeddings = OpenAIEmbeddings(
#             openai_api_key=st.session_state.get("OPENAI_API_KEY")
#         )  # type: ignore
        
#         flattened_embeddings = []
#         metadata_list = []
#         for doc in docs:
#             doc_text = doc.page_content
#             embeddings_list = embeddings.embed_query(doc_text)
#             for i, chunk_embedding in enumerate(embeddings_list):
#                 metadata = {"page": doc.metadata["page"], "chunk": i}
#                 metadata_list.append(metadata)
#                 flattened_embeddings.append(chunk_embedding)

#         return FAISS.from_embeddings(flattened_embeddings, metadata=metadata_list)




# @st.cache(allow_output_mutation=True, show_spinner=False)
# def embed_docs(docs: List[Document]) -> VectorStore:
#     """Embeds a list of Documents and returns a FAISS index"""

#     if not st.session_state.get("OPENAI_API_KEY"):
#         raise AuthenticationError(
#             "Enter your OpenAI API key in the sidebar. You can get a key at"
#             " https://platform.openai.com/account/api-keys."
#         )
#     else:
#         # Embed the chunks
#         embeddings = OpenAIEmbeddings(
#             openai_api_key=st.session_state.get("OPENAI_API_KEY")
#         )  # type: ignore
#         index = FAISS.from_documents(docs, embeddings)

#         print(index)

#         return index



@st.cache(allow_output_mutation=True)
def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    print(query)

    print(index)

    # Search for similar chunks
    doc_chunks = index.similarity_search(query, k=5)


    print(doc_chunks)

    return doc_chunks



@st.cache(allow_output_mutation=True)
def get_answer(doc_chunks: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0, openai_api_key=st.session_state.get("OPENAI_API_KEY")
        ),  # type: ignore
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {"input_documents": doc_chunks, "question": query}, return_only_outputs=True
    )
    return answer



@st.cache(allow_output_mutation=True)
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs

def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
