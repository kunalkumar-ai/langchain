import os
from langchain.agents import AgentExecutor, tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def get_time(request: str) -> str:
    """Returns the current time or date. Use input like 'time', 'date', or 'now'."""
    from datetime import datetime
    now = datetime.now()
    request = request.lower()
    if "date" in request:
        return now.strftime("%Y-%m-%d")
    elif "time" in request:
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_user_info(user: str) -> str:
    """Returns background info on a known person. Use when user asks about someone like 'Kunal', 'Gunjan', or 'Ada'."""
    users = {
    "kunal": "Kunal Kumar is an AI/ML engineer and student at Ulm University.",
    "gunjan": "Gunjan is studying Cognitive Systems and is interested in HCI.",
    "elon": "Elon Musk is the CEO of SpaceX and Tesla.",
    "ada": "Ada Lovelace is considered the first computer programmer."
    }

    return users.get(user.lower(), "No information found.")


# Step 2: Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step 3: Define prompt
tool_descriptions = "\n".join([
    "- get_time: Returns current date or time based on user's request.",
    "- get_user_info: Returns info about known people like 'Kunal', 'Gunjan', or 'Ada'."
])

prompt = ChatPromptTemplate.from_messages([
    (("system", 
 "You are an information provider. You can use tools to help answer questions, especially when the question is about dynamic, personal, or stored data.\n"
 "If no tool is appropriate, feel free to answer using your own knowledge.\n"
 "When using your own knowledge, make sure your response is clean, helpful, and well-formatted — no repetition, no unnecessary explanations.\n"
 "\nAvailable tools:\n" + tool_descriptions)
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])




# Step 4: Create agent with tool
tools = [ get_time, get_user_info]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 5: Run the agent
while True:
    user_input = input("\nAsk a question (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    response = agent_executor.invoke({"input": user_input})
    print("\nAgent Response:\n", response['output'])
