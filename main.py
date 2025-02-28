from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import revisor, first_responder
from tool_executor import tool_node
from typing import List

load_dotenv()

# make 2 iterations in critique and revision node
MAX_ITERATIONS = 2
builder = MessageGraph()
# responder
builder.add_node("draft", first_responder)
# execute tools
builder.add_node("execute_tools", tool_node)
# revisor
builder.add_node("revise", revisor)
# connect nodes and create edges with start key and end key
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# run after revisor node, to decide which node to go next (output or reiteration of tool execution)
def event_loop(state: List[BaseMessage]) -> str:
    # how many times we have iterated
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)

builder.set_entry_point("draft")

graph = builder.compile()
print(graph.get_graph().draw_ascii())
graph.get_graph().draw_mermaid_png(output_file_path="/Users/junfanzhu/Desktop/reflexion-agent/graph.png")
print(graph.get_graph().draw_mermaid())

if __name__ == '__main__':
    print("Hello Reflexion Agent!")
    res = graph.invoke(
        "Write about DeepSeek MoE and GRPO, list its impact and applications to future AI research."
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
    print(res)