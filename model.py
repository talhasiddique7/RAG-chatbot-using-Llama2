from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a helpful AI assistant. Use the context below to answer the user's question.  

Context: {context}  
Question: {question}  

- If the user asks for a joke, return the best joke from the context.  
- If no joke is found, politely say you don't know.  
- Do **not** avoid jokes unless they are offensive.  

Answer:
"""





def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context','question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 10}), return_source_documents=True, chain_type_kwargs={'prompt': prompt}, verbose=True)
    return qa_chain

#Loading the model
def load_llm():
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 50,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to RAG Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
