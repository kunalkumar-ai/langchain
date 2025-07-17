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
def get_user_from_file(name: str) -> str:
    """Looks up user info from a local JSON file called users.json. Use this for user-specific queries."""
    import json
    import os
    try:
        path = os.path.abspath("users.json")
        print(f"Looking for users.json at: {path}")
        with open(path, "r") as f:
            users = json.load(f)
        return users.get(name.lower(), "User not found.")
    except FileNotFoundError:
        return "users.json file not found."
    except Exception as e:
        return f"Error: {str(e)}"

# Step 2: Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step 3: Define prompt
tool_descriptions = "\n".join([
    "- get_time: Returns current date or time based on user's request.",
    "- get_user_from_file: Looks up a user's info from users.json."
])

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a knowledgeable AI assistant with access to both your own general knowledge and external tools.\n\n"
     "## TOOL USAGE:\n"
     "Use tools when the question is about dynamic, personal, or stored data that is not part of your general knowledge.\n"
     "- Use `get_user_from_file` if the user is asking about a specific person whose info may be stored locally.\n"
     "- Use `get_time` when the user asks for the current date or time.\n\n"
     "If the question can be answered from your own knowledge (e.g., general facts, concepts), feel free to respond directly.\n\n"
     "## RESPONSE GUIDELINES:\n"
     "- Be clear, helpful, and concise\n"
     "- Use markdown formatting when appropriate (e.g., for steps or lists)\n"
     "- Do not mention which tool you used\n"
     "- Say 'I don't know' if you truly can't find the answer\n"
     "- Never fabricate information\n\n"
     "Available tools:\n" + tool_descriptions
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])





# Step 4: Create agent with tool
tools = [ get_time, get_user_from_file]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 5: Run the agent
while True:
    user_input = input("\nAsk a question (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    response = agent_executor.invoke({"input": user_input})
    print("\nAgent Response:\n", response['output'])
