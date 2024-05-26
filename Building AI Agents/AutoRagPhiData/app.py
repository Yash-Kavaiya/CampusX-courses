from typing import List
import nest_asyncio
import logging
import os
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
import psycopg

os.environ['OPENAI_APT_KEY']='OPENAI_APT_KEY'

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
logger = logging.getLogger(__name__)

# 1. Setup Assistant
def setup_assistant(llm: str) -> Assistant:
    return Assistant(
        name="auto_rag_assistant",
        llm=llm,
        storage=PgAssistantStorage(table_name="auto_rag_assistant_openai", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_openai",
                embedder=OpenAIEmbedder(model="text-embedding-3-small",api_key='OPENAI_APT_KEY', dimensions=1536),
            ),
            num_documents=3,
        ),
        description="You are a helpful Assistant called 'AutoRAG' and your goal is to assist the user in the best way possible.",
        instructions=[
            "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the users question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        tools=[DuckDuckGo()],
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )

# 2. Add Document to Knowledge Base
def add_document_to_kb(assistant: Assistant, file_path: str, file_type: str = "pdf"):
    if file_type == "pdf":
        reader = PDFReader()
    else:
        raise ValueError("Unsupported file type")
    documents: List[Document] = reader.read(file_path)
    if documents:
        assistant.knowledge_base.load_documents(documents, upsert=True)
        logger.info(f"Document '{file_path}' added to the knowledge base.")
    else:
        logger.error("Could not read document")

# 3. Run Query
def query_assistant(assistant: Assistant, question: str):
    response = ""
    for delta in assistant.run(question):
        response += delta  # type: ignore
    return response

if __name__ == "__main__":
    nest_asyncio.apply()
    llm_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    llm=OpenAIChat(model=llm_model,api_key='OPENAI_APT_KEY')
    assistant = setup_assistant(llm)
    sample_pdf_path = "sample.pdf"
    add_document_to_kb(assistant, sample_pdf_path, file_type="pdf")
    query = "What is the main topic of the document?"
    response = query_assistant(assistant, query)
    print("Query:", query)
    print("Response:", response)