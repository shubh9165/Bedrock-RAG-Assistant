import boto3
import json
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM,ChatBedrockConverse


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

bedrock_clinet=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_clinet)

def data_ingestion():
    loader=PyPDFLoader("Attention.pdf")
    docs=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=text_splitter.split_documents(docs)
    return docs

def get_vector_store(docs):
    vectore_store=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectore_store.save_local("faiss_index")

def get_lamma_llm():
    # MODIFIED: Using ChatBedrockConverse for the Converse API
    return ChatBedrockConverse(
        model_id="meta.llama3-8b-instruct-v1:0",
        temperature=0.5,
        max_tokens=512,
        top_p=0.9,
        region_name="us-east-1",
        client=bedrock_clinet
    )

   




prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to provide a concise answer of at least 250 words. Context: {context}"),
    ("human", "{input}"),
])


def get_response_llm(llm,vectore_store,query):
    documnet_chain=create_stuff_documents_chain(llm,prompt)

    retriever = vectore_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )

    chain=create_retrieval_chain(retriever=retriever,combine_docs_chain=documnet_chain)
    response=chain.invoke({"input":query})
    return response["answer"]


def main():
    st.set_page_config("Bedrock RAG Assistant")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_qustion =st.text_input(label="Enter your question ")

    with st.sidebar:
        st.title("Update or Create vectore")

        if st.button("update vector"):
            st.spinner("Processing")
            docs=data_ingestion()
            get_vector_store(docs)
            st.success("Done")
    
    if st.button("Use Lamma"):
        st.spinner("Processing")
        vectore_store=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
        llm=get_lamma_llm()
        st.write(get_response_llm(llm=llm,vectore_store=vectore_store,query=user_qustion))
        st.success("Done")



if __name__=="__main__":
    main()