
<!-- TOC --><a name="reflexion-agent-iterative-llm-research-and-refinement"></a>
# LangGraph Reflection Agent: Iterative LLM Research and Refinement

The LangGraph project implements a "Reflexion Agent" designed to iteratively refine answers to user queries using a Large Language Model (LLM) and web search. It simulates a research process where an initial answer is generated, critiqued, and revised based on information gathered from web searches, all managed by a LangGraph workflow. The project leverages LangChain, OpenAI's GPT-4, and the Tavily Search API to automate research and improve the quality of generated content. It highlights the use of LangGraph for complex, multi-step LLM applications and addresses challenges related to dependency management, LLM output parsing, and tool integration.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=junfanz1/LangGraph-Reflexion-Agent&type=Date)](https://star-history.com/#junfanz1/LangGraph-Reflexion-Agent&Date)

<!-- TOC --><a name="contents"></a>
## Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [LangGraph Reflection Agent: Iterative LLM Research and Refinement](#langgraph-reflection-agent-iterative-llm-research-and-refinement)
  - [Star History](#star-history)
  - [Contents](#contents)
  - [1. Purpose of the Project](#1-purpose-of-the-project)
  - [2. Input and Output](#2-input-and-output)
  - [3. LLM Technology Stack](#3-llm-technology-stack)
  - [4. Challenges and Difficulties](#4-challenges-and-difficulties)
  - [5. Future Business Impact and Further Improvements](#5-future-business-impact-and-further-improvements)
  - [6. Target Audience and Benefits](#6-target-audience-and-benefits)
  - [7. Advantages and Disadvantages](#7-advantages-and-disadvantages)
  - [8. Tradeoffs](#8-tradeoffs)
  - [9. Highlight and Summary](#9-highlight-and-summary)
  - [10. Future Enhancements](#10-future-enhancements)
  - [11. Prerequisites](#11-prerequisites)
  - [12. Setup](#12-setup)
  - [13. Code Explanation](#13-code-explanation)
    - [`main.py`](#mainpy)
    - [`chains.py`](#chainspy)
    - [`tool_executor.py`](#tool_executorpy)
    - [`schemas.py`](#schemaspy)
  - [14. How it Works](#14-how-it-works)
    - [Class/Function Breakdown](#classfunction-breakdown)
  - [15. Crucial Function: `execute_tools()`](#15-crucial-function-execute_tools)
  - [16. Future Improvements](#16-future-improvements)
  - [17. Additional Considerations](#17-additional-considerations)
  - [Acknowledgements](#acknowledgements)

<!-- TOC end -->

<!-- TOC --><a name="1-purpose-of-the-project"></a>
## 1. Purpose of the Project

This project implements a "Reflexion Agent" that iteratively refines answers to user queries using a Large Language Model (LLM) and web search. It simulates a research process where an initial answer is generated, critiqued, and then revised based on additional information gathered from web searches. The agent uses LangGraph to manage the state and flow of the process.

<!-- TOC --><a name="2-input-and-output"></a>
## 2. Input and Output

**Input:**

* A user's question or query in natural language (e.g., "Write about DeepSeek MoE and GRPO, list its impact and applications to future AI research.").

**Output:**

* A refined answer to the user's query, including:
    * A detailed answer (approximately 250 words).
    * A reflection/critique of the answer.
    * References to supporting information.

<!-- TOC --><a name="3-llm-technology-stack"></a>
## 3. LLM Technology Stack

* **LangChain:** Used for building LLM applications, including prompt templating, output parsing, and tool usage.
* **LangGraph:** Used to create stateful, multi-actor applications by defining a graph of nodes and edges.
* **OpenAI GPT-4 Turbo Preview:** The LLM used for generating and refining answers.
* **Tavily Search API:** Used for web search to gather additional information.
* **Pydantic:** Used for data validation and schema definition.
* **Dotenv:** Used for loading environment variables.

<!-- TOC --><a name="4-challenges-and-difficulties"></a>
## 4. Challenges and Difficulties

* **Node Duplication:** Managing the state and flow of the LangGraph required careful attention to node naming and connections.
* **Dependency Management:** Ensuring all required libraries are installed and compatible.
* **Import Errors:** Resolving import errors related to `langgraph` and `pydantic` versions.
* **LLM Output Parsing:** Reliably parsing structured data from LLM outputs.
* **Tool Integration:** Properly integrating external tools like the Tavily Search API.
* **Deprecation Warnings:** Keeping up to date with LangChain updates and fixing deprecation warnings.

<!-- TOC --><a name="5-future-business-impact-and-further-improvements"></a>
## 5. Future Business Impact and Further Improvements

* **Enhanced Research Capabilities:** Automating research processes for various domains.
* **Improved Content Generation:** Producing high-quality, well-researched content.
* **Personalized Learning:** Creating adaptive learning systems that refine explanations based on user feedback.
* **Automated Report Generation:** Generating detailed reports with accurate citations.
* **Future Improvements:**
    * Implement more sophisticated critique mechanisms.
    * Integrate additional data sources and tools.
    * Add user feedback loops for continuous improvement.
    * Improve error handling and robustness.
    * Implement more complex graph structures.

<!-- TOC --><a name="6-target-audience-and-benefits"></a>
## 6. Target Audience and Benefits

* **Researchers:** Automate literature reviews and information gathering.
* **Content Creators:** Generate high-quality, well-researched articles and reports.
* **Students:** Enhance learning through iterative refinement of answers.
* **Developers:** Build intelligent applications with automated research capabilities.

**Benefits:**

* Increased efficiency in research and content generation.
* Improved accuracy and reliability of information.
* Automated refinement of answers based on feedback and new information.

<!-- TOC --><a name="7-advantages-and-disadvantages"></a>
## 7. Advantages and Disadvantages

**Advantages:**

* Automated research and refinement process.
* Integration of web search for up-to-date information.
* Use of LangGraph for managing complex workflows.
* Improved quality of generated content through iterative feedback.

**Disadvantages:**

* Dependency on external APIs (OpenAI, Tavily).
* Potential for errors in LLM output parsing.
* Complexity of managing LangGraph state and nodes.
* Requires careful prompt engineering for optimal performance.

<!-- TOC --><a name="8-tradeoffs"></a>
## 8. Tradeoffs

* **Speed vs. Accuracy:** Iterative refinement takes time but improves accuracy.
* **Cost vs. Performance:** Using powerful LLMs and search APIs can be costly.
* **Complexity vs. Flexibility:** LangGraph provides flexibility but adds complexity.
* **Automation vs. Control:** Automating research reduces manual effort but may require less manual oversight.

<!-- TOC --><a name="9-highlight-and-summary"></a>
## 9. Highlight and Summary

This project demonstrates the power of LangGraph and LLMs for building iterative research agents. It showcases how to combine LLM-generated content with external tools and automated feedback loops to produce high-quality, refined answers.

<!-- TOC --><a name="10-future-enhancements"></a>
## 10. Future Enhancements

* **User Feedback Integration:** Allow users to provide feedback directly to the agent.
* **Multi-Modal Input/Output:** Support images, audio, and other data types.
* **Knowledge Graph Integration:** Integrate knowledge graphs for more structured information retrieval.
* **Improved Critique Mechanisms:** Implement more sophisticated critique and reflection.
* **Dynamic Tool Selection:** Allow the agent to dynamically choose tools based on the query.

<!-- TOC --><a name="11-prerequisites"></a>
## 11. Prerequisites

* Python 3.10+
* Poetry (recommended) or pip
* OpenAI API key
* Tavily Search API key

<!-- TOC --><a name="12-setup"></a>
## 12. Setup

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```
2.  **Create a virtual environment:**
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set environment variables:**
    * Create a `.env` file in the project root.
    * Add your OpenAI and Tavily API keys:
        ```
        OPENAI_API_KEY=your_openai_api_key
        TAVILY_API_KEY=your_tavily_api_key
        ```
5.  **Run the main script:**
    ```bash
    python main.py
    ```

<!-- TOC --><a name="13-code-explanation"></a>
## 13. Code Explanation

<!-- TOC --><a name="mainpy"></a>
### `main.py`

* **Purpose:** Orchestrates the LangGraph workflow.
* **Functionality:**
    * Loads environment variables.
    * Defines the LangGraph using `MessageGraph`.
    * Adds nodes for the first responder, tool executor, and revisor.
    * Connects nodes with edges.
    * Defines a conditional edge for the event loop.
    * Sets the entry point and compiles the graph.
    * Invokes the graph with a user query.
* **Crucial Functionality:**
    * `MessageGraph` is used to define the flow of the application. The add\_node and add\_edge functions are vital for connecting the LLM chains. The event\_loop function is used to decide the next step in the graph.
* **Future Improvements:**
    * Implement more complex graph structures.
    * Add error handling and logging.

<!-- TOC --><a name="chainspy"></a>
### `chains.py`

* **Purpose:** Defines the LLM chains and prompts.
* **Functionality:**
    * Loads environment variables.
    * Initializes the OpenAI LLM and output parsers.
    * Defines prompt templates for the first responder and revisor.
    * Creates LLM chains using `ChatPromptTemplate` and `llm.bind_tools()`.
* **Crucial Functionality:**
    * `ChatPromptTemplate` is used to create prompts that are fed to the LLM. The `llm.bind_tools` function is used to specify the Pydantic schemas that the LLM should use to generate its responses.
* **Future Improvements:**
    * Experiment with different prompt templates and LLM models.
    * Add more sophisticated output parsing.

<!-- TOC --><a name="tool_executorpy"></a>
### `tool_executor.py`

* **Purpose:** Executes web search queries and formats the results.
* **Functionality:**
    * Loads environment variables.
    * Initializes the Tavily Search API.
    * Defines the `execute_tools` function to run search queries in parallel.
    * Formats the search results into `ToolMessage` objects.
* **Crucial Functionality:**
    * The `execute_tools` function is crucial for running the tavily search API. The use of the `ThreadPoolExecutor` allows for parallel running of the web searches.
* **Future Improvements:**
    * Add error handling for API requests.
    * Implement more sophisticated result filtering and summarization.

<!-- TOC --><a name="schemaspy"></a>
### `schemas.py`

* **Purpose:** Defines Pydantic schemas for data validation.
* **Functionality:**
    * Defines `Reflection`, `AnswerQuestion`, and `ReviseAnswer` classes.
* **Crucial Functionality:**
    * Pydantic schemas are used to define the structure of the data that is passed between the LLM chains. This ensures that the data is in the correct format.
* **Future Improvements:**
    * Add more detailed validation rules.
    * Define schemas for other data types.

<!-- TOC --><a name="14-how-it-works"></a>
## 14. How it Works

1.  The user provides a query: inputs a natural language question or request into the system (e.g., "Write about DeepSeek MoE and GRPO, list its impact and applications to future AI research."). This query is the starting point for the agent's research and refinement process.
2.  Initialization and Graph Execution: The `main.py` script initializes the LangGraph, which begins at the "first_responder" node.
3.  The "first_responder" node uses the `first_responder` chain from `chains.py` to generate an initial answer using the OpenAI LLM. This answer is formatted according to the `AnswerQuestion` schema.
4.  The output from the "first_responder" node is passed to the "execute_tools" node.
5.  The "execute_tools" node uses the `execute_tools` function from `tool_executor.py` to:
    * Extract search queries from the LLM's output.
    * Execute these queries in parallel using the Tavily Search API.
    * Format the search results into `ToolMessage` objects.
6.  The search results are passed to the "revise" node.
7.  The "revise" node uses the `revisor` chain from `chains.py` to refine the initial answer based on the search results and generate a critique. The refined answer is formatted according to the `ReviseAnswer` schema.
8.  The `event_loop` function in `main.py` determines the next step:
    * If the maximum number of iterations has been reached, the process ends.
    * Otherwise, the process returns to the "execute_tools" node for another iteration.
9.  The process repeats steps 5-8 until the maximum number of iterations is reached.
10. The final refined answer is returned as the output.

<!-- TOC --><a name="classfunction-breakdown"></a>
### Class/Function Breakdown

* **`main.py`:**
    * `MessageGraph`: Defines the stateful graph.
    * `add_node()`: Adds nodes to the graph.
    * `add_edge()`: Connects nodes with edges.
    * `event_loop()`: Determines the next node based on the state.
    * `compile()`: Compiles the graph.
    * `invoke()`: Executes the graph.
* **`chains.py`:**
    * `ChatOpenAI`: Initializes the OpenAI LLM.
    * `JsonOutputToolsParser`, `PydanticToolsParser`: Parses LLM outputs.
    * `ChatPromptTemplate`: Creates prompt templates.
    * `first_responder`: Generates the initial answer.
    * `revisor`: Refines the answer.
* **`tool_executor.py`:**
    * `TavilySearchAPIWrapper`, `TavilySearchResults`: Interfaces with the Tavily Search API.
    * `execute_tools()`: Executes search queries and formats results.
* **`schemas.py`:**
    * `Reflection`, `AnswerQuestion`, `ReviseAnswer`: Defines Pydantic data schemas.

<!-- TOC --><a name="15-crucial-function-execute_tools"></a>
## 15. Crucial Function: `execute_tools()`

The `execute_tools()` function in `tool_executor.py` is crucial for integrating external information into the LLM's responses. Here's a detailed elaboration:

* **Purpose:**
    * Extract search queries from the LLM's output.
    * Execute these queries using the Tavily Search API.
    * Format the search results into `ToolMessage` objects.
* **Functionality:**
    1.  **Extract Queries:**
        * The function receives the current state (a list of messages) and extracts the latest `AIMessage`, which contains the LLM's output.
        * It uses the `parser` (defined in `chains.py`) to parse the LLM's output and extract the search queries.
    2.  **Parallel Execution:**
        * It uses `concurrent.futures.ThreadPoolExecutor` to execute the search queries in parallel. This significantly speeds up the process, especially when multiple queries are involved.
        * `executor.map(tavily_tool.run, search_queries)` applies the `tavily_tool.run` method to each query concurrently.
    3.  **Result Mapping:**
        * It maps the search results back to the original queries and IDs to maintain the context.
    4.  **Format Results:**
        * It converts the mapped results into `ToolMessage` objects, which are then passed to the next node in the LangGraph.
* **Importance:**
    * This function allows the LLM to access up-to-date information from the web, which is essential for generating accurate and comprehensive answers.
    * Parallel execution ensures that the process is efficient, even when dealing with multiple search queries.

<!-- TOC --><a name="16-future-improvements"></a>
## 16. Future Improvements

* **Improved Error Handling:** Implement robust error handling for API requests and LLM output parsing.
* **Dynamic Tool Selection:** Allow the agent to dynamically choose tools based on the query and context.
* **Knowledge Graph Integration:** Integrate knowledge graphs for more structured information retrieval.
* **User Feedback Loops:** Implement mechanisms for users to provide feedback and refine the agent's behavior.
* **Multi-Modal Input/Output:** Extend the agent to handle images, audio, and other data types.
* **Advanced Critique Mechanisms:** Implement more sophisticated critique and reflection techniques.
* **Caching:** Implement caching for API responses to reduce costs and improve performance.
* **Logging and Monitoring:** Add logging and monitoring to track the agent's performance and identify potential issues.

<!-- TOC --><a name="17-additional-considerations"></a>
## 17. Additional Considerations

* **API Key Security:** Ensure that API keys are stored securely and not exposed in version control.
* **Cost Management:** Monitor API usage and implement cost control measures.
* **Prompt Engineering:** Experiment with different prompt templates to optimize the agent's performance.
* **Scalability:** Consider the scalability of the application when deploying it in a production environment.
* **Testing:** Implement unit and integration tests to ensure the agent's reliability.
* **Documentation:** Maintain clear and up-to-date documentation for the project.
* **Virtual Environments:** Always use virtual environments to manage project dependencies.
* **Code Quality:** Adhere to coding standards and best practices.
* **Version Control:** Use version control (e.g., Git) to track changes and collaborate with others.
* **Security:** Implement security best practices to protect the application and user data.

<!-- TOC --><a name="acknowledgements"></a>
## Acknowledgements

[Eden Marco: LangGraph-Develop LLM powered AI agents with LangGraph](https://www.udemy.com/course/langgraph)
