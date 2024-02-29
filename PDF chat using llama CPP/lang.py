from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain.chains import RetrievalQA
import gradio as gr


embeddings = HuggingFaceHubEmbeddings()

llm = HuggingFacePipeline.from_model_id(model_id="TheBloke/Mistral-7B-v0.1-AWQ",task="text-generation",device=0,pipeline_kwargs={"max_new_tokens":30})
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#llm = VLLM(model="TheBloke/Mistral-7B-v0.1-AWQ",  vllm_kwargs={"quantization": "awq"},trust_remote_code=True,  # mandatory for hf modelsmax_new_tokens=128,)
def RAG(file,query):
    loader=UnstructuredFileLoader(file)
    #loader = TextLoader(file)
    documents = loader.load()

    docs= text_splitter.split_documents(documents)
    db=Chroma.from_documents(docs,embeddings)
    
    #result=db.similarity_search(query)
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=db.as_retriever(), 
                                    input_key="question")
    final=chain.invoke(query)
    #final =wrap_text_preserve_newlines(str(docs[0].page_content))
    
   
     
    return final['result']

iface = gr.Interface(
    fn=RAG, 
    inputs=["file","text"],
    outputs="text",
    title="Mistral chatbot",
).launch()