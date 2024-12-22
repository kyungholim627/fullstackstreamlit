from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
import os
import json


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

def get_questions_prompt(difficulty):
    difficulty_text = {
        "Easy": "Make an EASY and SIMPLE quiz that has TEN questions, asking about most major topics about {context}",
        "Normal": "Make a MODERATE DIFFICULTY quiz that has TEN questions, asking about major, essential details about {context}",
        "Hard": "Make a HARD, COMPLICATED, EASY TO BE WRONG quiz that has TEN questions, asking about minor, unessential details about {context}",
    }
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You are a helpful assistant role-playing as a teacher.
                {difficulty_text[difficulty]}
                """,
            )
        ]
    )

@st.cache_data(show_spinner = "Making quiz...")
def run_quiz_chain(_docs, diff_choice, topic):
    chain = {"context": format_docs} | questions_prompt | llm
    return chain.invoke(_docs)
    
with st.sidebar:
    key = st.text_input("Insert your OPEN AI KEY here.", type="password")

if key:
    os.environ["OPENAI_API_KEY"] = key
    with st.sidebar:
        docs = None
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)

    if not docs:
        st.markdown(
            """
        Welcome to QuizGPT!
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
        )
    else:
        function = {
    "name": "create_quiz",
    "description": "function that takes a list of TEN questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )
        diff_choice= st.selectbox(
        "Choose the difficulty of the quiz",
        (
            "Easy",
            "Normal",
            "Hard"
        ),
        )
        questions_prompt = get_questions_prompt(diff_choice)
        response = run_quiz_chain(docs, diff_choice, topic if topic else file.name)
        with st.form("questions_form"):
            if response and hasattr(response, "additional_kwargs"):
                arguments = response.additional_kwargs["function_call"]["arguments"]
                parsed_arguments = json.loads(arguments) 
                questions = parsed_arguments.get("questions", [])  
                correct_count = 0
                for idx, question in enumerate(questions):
                    value = st.radio(
                        f"{idx+1}: {question['question']}",
                        [
                            answer['answer'] for answer in question["answers"]
                        ],
                        index=None,
                    )
                    is_correct = False
                    for answer in question["answers"]:
                        if answer["answer"] == value and answer["correct"]:
                            is_correct = True
                            break
                    if is_correct:
                        st.success("Correct!")
                        correct_count += 1
                    elif value is not None:
                        st.error("Wrong!")
                button = st.form_submit_button()
                
        if button:
            if correct_count == len(questions):
                st.balloons()
                st.success("🎉 You got all answers correct! Well done!")
                retry_quiz = False  
            else:
                st.warning("❌ You didn't get all answers correct. Try again!")
                if st.button("Retry"):
                    st.experimental_rerun()  
else:
    st.markdown(
    """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by inserting your OPEN AI KEY <-
    """
    )

