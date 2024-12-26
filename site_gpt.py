from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

@st.cache_data
def load_wesite(url):
    loader = SitemapLoader(
    url,
    filter_urls =[
        r"(.*\/ai-gateway\/).*",
        r"(.*\/vectorize\/).*",
        r"(.*\/workers-ai\/).*"
    ],
    parsing_function = parse_page
    )
    docs = loader.load()
    return docs

def parse_page(soup):
    header = soup.find("header")
    menu = soup.find('ul', class_='top-level astro-3ii7xxms').find_all('li', class_='astro-3ii7xxms')
    if header:
        header.decompose()
    if menu:
        for li in menu:
            li.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

with st.sidebar:
    key = st.text_input("Insert your OPEN AI KEY here.", type="password")
    
if key:
    os.environ["OPENAI_API_KEY"] = key
    with st.sidebar:
        url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )
    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a Sitemap URL.")
        else:
            docs = load_wesite(url)
            st.write(docs)