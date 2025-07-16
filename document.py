import os
from langchain.agents import AgentExecutor, tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# Step 1: Define a simple tool
@tool
def load_document_info(doc_name: str) -> str:
    """Load document info from a simple simulated store."""
    docs = {
        "project": "This project is a demo of LangChain agents using Gemini.",
        "author": "Kunal Kumar is the creator of this AI agent.",
        "langchain": "LangChain is a framework for building LLM applications."
    }
    return docs.get(doc_name.lower(), "Sorry, I don't have that document.")

# Step 2: Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step 3: Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # ✅ Required for tool-using agents
])

# Step 4: Create agent with tool
tools = [load_document_info]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 5: Run the agent
while True:
    user_input = input("\nAsk a question (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    response = agent_executor.invoke({"input": user_input})
    print("\nAgent Response:\n", response['output'])
