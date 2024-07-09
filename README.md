# Multimindbot 

Multimindbot is a chatbot that makes it easy for users to find content or key dates without having to search the entire website. Users simply need to ask their query and the bot will provide the answers. Multimindbot is designed specifically for the college admission counseling process.
<hr>

**Used technology in the project:**
1) Langchain
2) LLM (Llama-2-7B-Chat-GGUF)
3) Prompt Engineering
4) Huggingface
5) streamlit

**Final output of the model :** (English model)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/c5f4f9bb-3cc2-4a11-85fd-e171e323a25c)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/aff04e40-5bd4-4472-83f6-02b514905579)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/9741fb7e-409a-4d7e-b3c4-dd96f8fffabb)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/30c3636d-50ae-43de-82a5-cd672b1da776)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/5db9b1e7-f635-43f2-9db5-e0706607dcec)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/7e8fbcba-8529-427e-ab6a-c1b23cf94044)

![image](https://github.com/neha13rana/Chatbot/assets/121093178/ec4313ae-9b9e-41f3-999e-a20821d76906)

**Question - Answer with chathistory :** (With streamlit localhost)

![image](https://github.com/neha13rana/Multimindbot/assets/121093178/b799dea6-c71a-4e8f-baca-c039071be86f)

![image](https://github.com/neha13rana/Multimindbot/assets/121093178/342d068b-9a43-4a23-a987-0a01b46b0047)

![image](https://github.com/neha13rana/Multimindbot/assets/121093178/727ce6c9-06f5-41bd-afb8-5ab6e87c7a5f)

![image](https://github.com/neha13rana/Multimindbot/assets/121093178/8720a2ec-4b1d-4fe8-bdc7-1676a75ca63b)

![WhatsApp Image 2024-07-03 at 18 50 28_2e9a2957](https://github.com/neha13rana/Multimindbot/assets/121093178/3af1d11f-78a9-4633-9716-21421ac88c98)


<hr>

**Model Architecture :** 

![langchain1 drawio](https://github.com/neha13rana/Chatbot/assets/121093178/5a0de972-1ebb-4381-b09a-0b74255bef54)

<hr>

**About the model development :**

1)	Data collection :
Gather all the relevant data of the LLM application in our case we collected data from the ACPC and the ACPDC sites and made question-and-answer pairs in the documents.( https://gujacpc.admissions.nic.in/ ,https://gujdiploma.admissions.nic.in/)

2)	Data cleaning: Cleaning the sentences and the text in the document for accurate result.


3)	RAG -Langchain :
1. Data loader: loaders deal with the specific of accessing and converting the data.
(In our case it is a csv file and other files related to the acpc and acpdc.)

2. Document splitting: splitting the main document into several smaller chunks (for retaining meaningful relationships). There are many types of document splitter but we use a character splitter that looks at characters.

3. Embedding: Embedding vector captures meaning, text with similar content will have similar vectors.
4. Vector Store
5. Retriever: for accessing/ indexing the data in the vector store (we use similarity among the 3 types check for MMR)
6. Prompt: retrievalQA chain, takes the question from the user and passes to the LLM. (stuff chain type - It takes a list of documents, inserts them all into a prompt, and passes that prompt to an LLM. This chain is well-suited for applications where documents are small and only a few are passed in for most calls.
7. LLM: LLMs are advanced AI systems that can perform tasks such as language translation, text completion, summarization, and conversational interactions. They work by analyzing large amounts of language data and are pre-trained on vast amounts of data.

Chathistory, memory: 
Conversationalbuffermemory: keeps a list, and buffer of chat messages in history and it’s going to pass those along with the question.

**Deliverables :**

1. Get all the latest updates, admission details, and counseling support in one place, so you won't have to search through multiple links.
2. Save time by avoiding the hassle of browsing through various sources for relevant information.
3. Choose your preferred language for seamless communication, ensuring suitability and ease of understanding.
4. Receive quick and effective support for admission and counseling queries, enhancing productivity.

<hr>

**Steps to use this website :**
1) Download the project folder.
2) set the environment in your code editor's CMD by writing  !(python -m venv venv) then write  !(venv\Scripts\activate)
if it is installed. 
3) install requirements.txt by writing !(pip install -r requirements.txt)
4) then write this command to run the app !(streamlit run app.py)

