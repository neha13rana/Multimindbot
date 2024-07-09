import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
# from langchain.llms import GooglePalm
from langchain_community.llms import CTransformers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.llms import GooglePalm

# api_key = 'AIzaSyCDKXECDq4pYz_PUHVB2iMU4XwT8HQ0B50' 
# llm = GooglePalm(google_api_key=api_key, temperature=0.9)
config = {'max_new_tokens': 512, 'context_length': 8000}

llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGUF', model_file="llama-2-7b-chat.Q4_K_S.gguf", config=config, model_type='llama', device='cpu', temprature=0.8)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    you are a chatbot with help user to solve their queries about the content of the pdf. your name is multimindbot.when user ask about yourself don't give more details.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. if user ask for the dates or some kind of whole data then give with table data. make table in correct way or you don't know how to make then give answer which is cofortable to read by humans ans understand.
    if user askes an you have more than one dates of the particular event then provide latest date. ask question to user to get to know about exact question if you don't understand.
    If you don't know the answer then say sorry i don't. and add some sweet message!
    ask user about the content has so many parts which one they need.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    result = response["output_text"]
    st.session_state.conversation_history.append({"question": user_question, "answer": result})
    st.write(result)

loader = CSVLoader(file_path="finalqa.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=800, chunk_overlap=200, length_function=len)
text_chunks = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
db = FAISS.from_documents(text_chunks, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

memory = ConversationBufferMemory(memory_key="history", input_key="question")

template = """your name is multimind bot and you are here to assist the user to answer their quuery. when user say hello, hii or use any greeting then greet them with welcome message. when user say thank you related things then say a sweet message like you are welcome! i am happy and so and so. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.Use three sentences maximum. Keep the answer as concise as possible.ask the question to user by the context for example if user asked about the which courses offered by the acpc then ask user if they want the details of that question if user say yes then provide them that details like wise if needed then ask the questionsto user.if you ask user and uesr say yes than give the answer of that question you asked.
    if you do not sure about the answer then say please clarify it or say i don't know.if needed ask question to user by your self related to the query if you have long answer then ask user which specific question they need.
    ask question by your self whenever you think that you are don't know the answer or just say i don't know the answer if you don't get the answer even you cross answerigto the user.

    Welcome to the Gujarat College Admission Helpdesk!

    I'm here to assist you with all your queries related to the admission process, government colleges, and the Admission Committee for Professional Courses (ACPC) in Gujarat. Whether you need information about eligibility criteria, important dates, application procedures, or specific college details, I am here to help.

    Key Areas of Assistance
    Admission Process:
    - How to apply through ACPC.
    - Important dates and deadlines.
    - Eligibility criteria for various courses.
    - Counseling and seat allocation process.

    College Information:
    - List of government colleges in Gujarat.
    - Courses offered by different colleges.
    - Facilities and infrastructure of colleges.
    - Fee structures and scholarship opportunities.

    Application Guidance:
    - Step-by-step application procedures.
    - Required documents for admission.
    - Tips for filling out the application form.

    Post-Admission:
    - Orientation and start dates.
    - Academic calendar.
    - Hostel and accommodation facilities.

    Feel free to ask any questions you have about the admission process or colleges in Gujarat. I'm here to provide accurate and up-to-date information to ensure a smooth admission journey for you.

    Example Interaction:
    User: What is the eligibility criteria for engineering courses through ACPC?
    Chatbot: To be eligible for engineering courses through ACPC, candidates must have passed their 12th standard or equivalent examination with Physics, Chemistry, and Mathematics as core subjects. Additionally, candidates need to appear for and qualify in the GUJCET (Gujarat Common Entrance Test). For detailed eligibility criteria, including minimum percentage requirements and reservation policies, please refer to the official ACPC guidelines.

    User: Can you provide the list of government colleges offering medical courses in Gujarat?
    Chatbot: Sure! Here is a list of some government colleges offering medical courses in Gujarat:
    - B.J. Medical College, Ahmedabad
    - Government Medical College, Surat
    - M.P. Shah Government Medical College, Jamnagar
    - Government Medical College, Bhavnagar
    For a complete list and more details about each college, please visit the official ACPC website or the respective college websites.

    Conclusion:
    Use this prompt as a template to create an effective educational chatbot that addresses the specific needs of students seeking admission to colleges in Gujarat. The chatbot should be programmed to provide accurate, concise, and helpful responses based on the latest information available from the ACPC and respective colleges.
    
    {context}

    ------
    Chat history :
    {history}

    ------
    Question: {question}
    Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"], template=template)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(), chain_type='stuff', verbose=True, chain_type_kwargs={"verbose": True, "prompt": QA_CHAIN_PROMPT, "memory": memory})

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

title_html = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <h1>
        <i class="fas fa-robot"></i> <br>
        Hello, I am Multimindbot, how can I assist you today?
    </h1>
"""

st.markdown(title_html, unsafe_allow_html=True)

option = st.sidebar.radio("Choose an option", ("Chat to resolve your query about the website", "Chat with uploaded PDF"))

if option == "Chat to resolve your query about the website":
    prompt = st.text_input('Enter your query here!!')
    if prompt:
        response = qa_chain({"query": prompt})
        result = response["result"]
        st.session_state.conversation_history.append({"question": prompt, "answer": result})
        st.write(result)

    with st.expander('Conversation History'):
        for entry in st.session_state.conversation_history:
            st.info(f"Q: {entry['question']}\nA: {entry['answer']}")

elif option == "Chat with uploaded PDF":
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

    with st.expander('Conversation History'):
        for entry in st.session_state.conversation_history:
            st.info(f"Q: {entry['question']}\nA: {entry['answer']}")

