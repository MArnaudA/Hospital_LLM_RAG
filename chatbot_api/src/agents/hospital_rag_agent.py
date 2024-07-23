import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import (
    Tool,
    AgentExecutor
)
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain import hub
from chains.hospital_review_chain import reviews_vector_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

prompt = hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]



chat_model = ChatGoogleGenerativeAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm_with_tools = chat_model.bind(tools=tools)


agent = (
    {
        "input": lambda x: x["input"]
    }
    |prompt
    |llm_with_tools
    |OpenAIFunctionsAgentOutputParser()
)

hospital_rag_agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True).with_types(
                                   input_type=HospitalQueryInput,
                                   output_type=HospitalQueryOutput
                               )