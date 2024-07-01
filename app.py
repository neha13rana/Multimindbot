from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st 
loader=CSVLoader(file_path="finalqa.csv" , encoding="utf-8" , csv_args={'delimiter': ','})
data=loader.load()
text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len)
text_chunks=text_splitter.split_documents(data)
embeddings= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device': 'cpu'},encode_kwargs = {'normalize_embeddings': True})
db =FAISS.from_documents(text_chunks,embeddings)
retriever =db.as_retriever(search_type="similarity", search_kwargs={"k":1})
api_key='AIzaSyBoCW3JQBd9pKt55AOJBqqMQCeEfP6lcRo'
llm =GooglePalm(google_api_key=api_key,temperature=0.9)
memory = ConversationBufferMemory(memory_key="history", input_key="question")
template="""your name is multimind bot and you are here to assist the user to answer their quuery. when user say hello, hii or use any greeting then greet them with welcome message. when user say thank you related things then say a sweet message like you are welcome! i am happy and so and so. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.Use three sentences maximum. Keep the answer as concise as possible.ask the question to user by the context for example if user asked about the which courses offered by the acpc then ask user if they want the details of that question if user say yes then provide them that details like wise if needed then ask the questionsto user.if you ask user and uesr say yes than give the answer of that question you asked.
    if needed ask question to user by your self related to the query if you have long answer then ask user which specific question they need.
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


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"],template=template,)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=db.as_retriever(),
                                       chain_type='stuff',
                                       verbose=True,
                                       chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT, "memory": memory,})

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("Hello, my name is Multimindbot how can i assist you today")
print("\n")
prompt = st.text_input('Enter your query here!! ')

if prompt:
    response = qa_chain({"query": prompt})
    result = response["result"]
    
    # Update conversation history in session state
    st.session_state.conversation_history.append({"question": prompt, "answer": result})
    
    st.write(result)

# Display conversation history
with st.expander('Conversation History'):
    for entry in st.session_state.conversation_history:
        st.info(f"Q: {entry['question']}\nA: {entry['answer']}")