from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


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
        .replace("                 Products  Learning  Status  Support  Log in   GitHub X YouTube     Select theme   DarkLightAuto                ", " ")
        .replace("\xa0", " ")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
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
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


with st.sidebar:
    key = st.text_input("Insert your OPEN AI KEY here.", type="password")
    
if key:
    os.environ["OPENAI_API_KEY"] = key
    st.markdown(
        """
        # SiteGPT
                    
            Ask questions about the content of a website.
                    
            Insert your sitemap URL.
        """
    )
    with st.sidebar:
        url = st.text_input(
        "Write down a URL in a .xml format",
        placeholder="https://example.com.xml",
    )
    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a Sitemap URL.")
        else:
            retriever = load_website(url)
            query = st.text_input("Ask a question to the website.")
            if query:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                result = chain.invoke(query)
                st.markdown(result.content.replace("$", "\$"))
else:
    st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by typing in your OPEN AI KEY <-
"""
)