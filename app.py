import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import os

# genai.configure(api_key="YOUR API KEY")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"YOUR GOOGLE APPLICATION CREDENTIALS PATH"

def get_text(docs):
    text=""
    for pdf in docs:
        pdf_data=PdfReader(pdf)
        for page in pdf_data.pages:
            text=text+page.extract_text()
    return text

def get_embeddings(text):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectors=FAISS.from_texts([text],embedding=embeddings)
    vectors.save_local('FAISS_index')

def get_response(question):
   db=FAISS.load_local('FAISS_index',GoogleGenerativeAIEmbeddings(model='models/embedding-001'),allow_dangerous_deserialization=True)
   docs=db.similarity_search(question)
   model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')
   prompt=PromptTemplate(template='Context:{context}\nQuestion:{question}',input_variables=['context','question'])
   chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)
   response=chain({'input_documents':docs,'question':question},return_only_outputs=True)
   st.write(response['output_text'])

def main():
    st.title("Chat With Documents")
    question=st.text_input("Enter Your Question")
    if st.button("Submit"):
        get_response(question)
        # pass

    with st.sidebar:
        st.title("Upload Document Here")
        documents=st.file_uploader("", type=['pdf'],accept_multiple_files=True)
        if st.button("Upload"):
            text=get_text(documents)
            get_embeddings(text)
            st.success("File Uploaded Successfully")

if __name__ == "__main__":
    main()
