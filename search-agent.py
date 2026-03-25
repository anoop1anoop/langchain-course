import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


# Implementation using the TavilyClient
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches the internet about a topic and return the results.
    Arguments:
        query: The topic to search about.
    Returns:
        A string with the search results.
    """ 
    print(f"Searching for {query}\n")
    return tavily.search(query=query)


class Source(BaseModel):
    '''Schema for a source used by the agent'''
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    '''Schema for the agent answer and responses'''
    answer: str = Field(description="The answer to the user's question")
    sources: List[Source] = Field(default_factory=list,description="The sources used to answer the question")


#llm = ChatOllama(model="llama3.2:latest")
llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!!!\n")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postigs for engineer using langchain in the bay area on linkedin and list their details and sources")})
    print(result)


if __name__ == "__main__":
    main()

