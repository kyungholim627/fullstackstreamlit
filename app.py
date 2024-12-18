import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
import os

# Initial Status
if "messages" not in st.session_state:
    st.session_state["messages"]=[]

# Define Variables


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages= True,
)


def split_embed_file(file):
    file_content=file.read()
    file_path=f"./cache/files/{file.name}"
    with open(file_path,"wb") as f:
        f.write(file_content)
    cache_dir=LocalFileStore(f"./cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,  
        chunk_overlap=100,
        separator="\n",
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedding = OpenAIEmbeddings()
    cached_embeddings=CacheBackedEmbeddings.from_bytes_store(
        embedding, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever



#Streamlit function
def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message,role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def message_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


# Streamlit 
st.set_page_config(page_title="Challenge GPT")
st.title("Challenge GPT")
st.markdown("""
            Welcome! 
            
            Use this chatbot to ask questions about your files to your AI!
            """)



with st.sidebar:
    key = st.text_input("Insert your OPEN AI KEY here.", type="password")
    
if key:
    os.environ["OPENAI_API_KEY"] = key
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4",
        )
    with st.sidebar:
        file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["pdf","txt", "docx"])
    if file:
        message = st. chat_input("Ask anything about your file.")
        retriever=split_embed_file(file)
        send_message("It's all set!", "ai", save=False)
        if message:
            message_history()
            send_message(message, "human")
            map_doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text VERBATIM. DO NOT tell me any other details. DO NOT answer the question. DO NOT change the original text. JUST return the portions of the text that seems relevant.
            ------------------
            {portion}
            """),
            ("human", "{question}"),
            ])
            map_doc_chain = map_doc_prompt | llm
            
            def map_docs(inputs):
                documents = inputs["documents"]
                question = inputs["question"]
                return "\n\n".join(map_doc_chain.invoke({
                    "portion": doc.page_content,
                    "question": question
                }).content for doc in documents)

            map_chain = {"documents": retriever, "question": RunnablePassthrough()} | RunnableLambda(map_docs)

            final_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                Give the following extracted parts of a long document and a question, create a final answer.
                If you don't know the answer, just say that you don't know. Don't try to make up the answer.
                ---------
                {context}
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            def load_memory(input):
                return memory.load_memory_variables({})["chat_history"]

            chain = {"context": map_chain, "question": RunnablePassthrough()} | RunnablePassthrough.assign(chat_history=load_memory) | final_prompt | llm

            def invoke_chain(question):
                result=chain.invoke(question)
                memory.save_context(
                    {"input": question},
                    {"output": result.content}
                )
                return result.content
            response = invoke_chain(message)
            send_message(response, "ai")
    else:
        st.session_state["messages"] = []

else:
    st.markdown("""         
            <- Insert OPEN AI KEY first!
            """)


