import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import gradio as gr

import asyncio

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_model = os.getenv("OPENAI_CHAT_MODEL")
embeddings = os.getenv("OPENAI_EMBEDDING_MODEL")

with open("data/evaluation.txt", "r", encoding="UTF-8") as file:
    data_eval = file.read()

with open("data/knowledge.txt", "r", encoding="UTF-8") as file:
    data_know = file.read()


chunks = [
    chunk.strip()
    for chunk in data_know.split("\n\n")
    if chunk.strip()
]

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)


vector_store = FAISS.from_documents(
    documents,
    embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)


llm = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL"),
    temperature=0.3
)


def rag_run(message):

    retrieved_docs = retriever.invoke(message)

    retrieved_contexts = [
        doc.page_content
        for doc in retrieved_docs
    ]

    context = "\n\n".join(retrieved_contexts)

    prompt = f"""
You are a helpful assistant.
Answer the general questions like a human having a normal conversations, but do not reply to questions 
that are completely off the topic from the given contexts. 

General Questions Examples: hi, hello, i don't know this, i don't like you, and many more. 
1. I want you to act like human 
2. have general conversations but do not go off topic 
3. for the questions based on the topic answer only through topic. but general conversations use your own brain
4. unless asked do not go off topic
5. do not reply to questions that are completely off the topic from the given contexts.
6. reply to general questions like hi, hello, i don't know this, i don't like you, and many more.

Example Question: is Virat Kohli the best cricketer in the world?
Answer: I don't know this. Please ask me about ERR-9942.

7.follow this for the questions that are simialr to this. do not answer those types of confusing questions.
8.for general conversations use your own brain but do not go off topic too much (a little bit off topic is allowed)

Question:
{message}

Context:
{context}
"""

    response = llm.invoke(prompt)
    return response.content, retrieved_contexts


def chatbot(message, history=None):
    

    response, retrieved_contexts = rag_run(message)

    return response


demo = gr.ChatInterface(
    fn=chatbot,
    title="AI RAG Chatbot"
)




def read_evaluation_data():
    
    question = data_eval.split("USER_QUERY:")[1].split("RELEVANT_CONTEXT:")[0].strip().strip('"')
    context = data_eval.split("RELEVANT_CONTEXT:")[1].split("TARGET_RESPONSE:")[0].strip().strip('"')
    target_response = data_eval.split("TARGET_RESPONSE:")[1].strip().strip('"')
    
    return question, context, target_response

async def run_ragas():
    
    question, context, target_response = read_evaluation_data()
    answer, retrieved_contexts = rag_run(question)
    
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        reference_contexts=[context],
        retrieved_contexts=retrieved_contexts,
        reference=target_response
    )
    
    client = AsyncOpenAI(
        api_key=api_key
    )

    evaluator_llm = llm_factory(
        api_model,
        client=client
    )

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=embedding_model
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall = ContextRecall(
        llm=evaluator_llm
    )

    faithfulness_score = await faithfulness.single_turn_ascore(sample)

    answer_relevancy_score = await answer_relevancy.single_turn_ascore(sample)

    context_precision_score = await context_precision.single_turn_ascore(sample)

    context_recall_score = await context_recall.single_turn_ascore(sample)
    
    print(f"Faithfulness: {faithfulness_score}")
    print(f"Answer Relevancy: {answer_relevancy_score}")
    print(f"Context Precision: {context_precision_score}")
    print(f"Context Recall: {context_recall_score}")
    
choice = input("Enter 1 for Chatbot or 2 for RAGAS: ")

if choice == "1":
    demo.launch()

elif choice == "2":
    asyncio.run(run_ragas())