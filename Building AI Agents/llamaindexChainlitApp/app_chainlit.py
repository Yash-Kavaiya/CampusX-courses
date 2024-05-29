import json
from typing import Sequence, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import nest_asyncio
import chainlit as cl
from llama_index.llms.groq import Groq

nest_asyncio.apply()
#llm = Ollama(model="phi3", request_timeout=120.0)
llm =Groq(model="mixtral-8x7b-32768", api_key="api_key")
Settings.llm = llm



def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm,
    verbose=True,
)

####response=agent.chat('what is (65*7)/8')
###print(str(response))

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello there! I am your Agent How can I help YOU?").send()
    cl.user_session.set('agent',agent)    

@cl.on_message
async def on_message(message:cl.Message):
    agent=cl.user_session.get('agent') 
    response=agent.chat(message.content)
    await cl.Message(content=str(response)).send()
