import json
import gradio as gr


from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



#Create the prompte from the template.
promptTemplate = """Answer the question as precise as possible using the provided context. If the answer is
    not contained in the context, say "answer not available in context" \n\n
    Context: {context}
    Question: {question}
    Answer:

     """
modelSel = ""

#Load the PDF file to ChromaDB
#输入数据
def loadDataFromPDFFile(filePath):
    loader = PyPDFLoader(filePath) #载入PDF文件
    pages = loader.load_and_split() #split PDF文件
    chunks = filter_complex_metadata(pages) #过滤掉metadata，无法被存储和理解
    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())#存入chromadb变为vector database
    return vector_store #返回数据库


#retriever的创建
def modelResponse(message , history):
    llm = ChatOllama(model = conf["model"])

    prompt = PromptTemplate(template=promptTemplate , input_variables=["context","question"])

    #Initiate the retriever
    dbLoaded = loadDataFromPDFFile("~/Desktop/report.pdf")
    retriever = dbLoaded.as_retriever(search_type="similarity_score_threshold" , search_kwargs = {
        "k": 3, 
        "score_threshold": 0.2 
    })
    hpChain = (#Langchain的模型 LCEL
            {"context": retriever , "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return hpChain.invoke(message) #message是我传入的相应的query


if __name__ == "__main__":

    #read configuration file
    conf = {}
    with open("config.json" , "r") as confFile:
        conf = json.load(confFile)
        print(conf["model"])

    chatUI = gr.ChatInterface(fn=modelResponse , title="Tsoi Tsoi Q&A")
    chatUI.launch()
