# 🛠️ LangGraph Multi-Tool Agent & Local FAISS RAG

A stateful AI agent built with **LangGraph** and **LangChain** that combines **local offline RAG** (using free Hugging Face `SentenceTransformers` and `FAISS`) with **12 custom tools** for math, intent/keyword extraction, live web search, weather, currency/unit conversion, and file management.

Managed end-to-end with **[uv](https://docs.astral.sh/uv/)** for fast, reproducible Python dependency management.

---

## ✨ Features

* **Stateful LangGraph Workflow:** Uses `StateGraph` and a custom `AgentState` (`MessagesState`) to orchestrate multi-step tool calling (`llm_call` ⇄ `tool_node`) and track total LLM invocations.
* **100% Free Local Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` to index and search local documents in FAISS without embedding API costs or rate limits.
* **12 Production-Ready Custom Tools:** Built with `@tool`, strict Python type hints, and Pydantic validation so the LLM can chain multiple tools in a single turn.
* **Fast Dependency Management:** Configured for `uv` (`pyproject.toml` + `uv.lock`).

---

## 🧰 Included Custom Tools

| # | Tool Name | Description |
| :--- | :--- | :--- |
| 1 | `calculator` | Safely evaluates mathematical expressions (`+`, `-`, `*`, `/`, `**`) via Python AST. |
| 2 | `extract_intent_and_keywords` | Classifies user intent (`question`, `complaint`, `booking`, `greeting`, `search`, `calculation`, `other`) and extracts keywords as structured JSON. |
| 3 | `greet_user` | Generates a time-appropriate personalized greeting (`Good morning/afternoon/evening`). |
| 4 | `tavily_web_search` | Fetches real-time web search results and news using the Tavily API. |
| 5 | `get_current_datetime` | Returns the current system date, day of the week, and time. |
| 6 | `get_live_weather` | Fetches live city weather conditions via `wttr.in` (no API key required). |
| 7 | `search_wikipedia` | Retrieves concise encyclopedia summaries from Wikipedia. |
| 8 | `convert_currency` | Converts amounts between international currencies (e.g., `USD` to `NPR`, `EUR` to `INR`) using live exchange rates. |
| 9 | `convert_units` | Converts distance (`km`/`miles`), weight (`kg`/`lbs`), and temperature (`°C`/`°F`). |
| 10 | `analyze_text_stats` | Counts exact characters, words, and sentences in a text block. |
| 11 | `extract_contacts_from_text` | Uses regex to extract email addresses and phone numbers from unstructured text. |
| 12 | `save_note` | Safely writes notes, reports, or summaries to a local file. |
| + | `search_knowledge_base` | Queries the local `FAISS` vector store built with `all-MiniLM-L6-v2` embeddings. |

---

## 📁 Project Structure

```text
├── ragwithlanggraph.ipynb             # Definitions for all 12 custom @tool functions, LangGraph StateGraph (llm_call, tool_node, conditional edges) ,Script to chunk documents and save the local FAISS index
├── faiss_index/         # Saved local FAISS vector store directory
├── .env                 # Environment variables (API keys)
├── pyproject.toml       # Project dependencies managed by uv
└── README.md            # Project documentation
```

---

## 🚀 Quickstart (with `uv`)

### 1. Prerequisites
* **Python 3.11+**
* **[uv](https://docs.astral.sh/uv/)** installed:
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
  ```

### 2. Install Dependencies
Initialize and sync all project packages through requirements.txt

### 3. Configure Environment Variables
Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

---

## 🧠 Building the Local FAISS Vector Index (Optional RAG)

To create your local vector store using the free Hugging Face model (`sentence-transformers/all-MiniLM-L6-v2`):

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_DIR)
print(f"Index saved to {INDEX_DIR}/")
```

---

## 🔄 How the LangGraph Workflow Operates

```text
          +---------+
          |  START  |
          +----+----+
               |
               v
        +------------+
        |  llm_call  |<---------+
        +------+-----+          |
               |                |
               v                |
       (should_continue)        |
          /         \           |
  tool_calls?      no tool_calls|
        /             \         |
       v               v        |
+-------------+     +-----+     |
|  tool_node  |---->| END |     |
+------+------+     +-----+     |
       |                        |
       +------------------------+
```

1. **`llm_call` Node:** Sends the `SystemMessage` and conversation history (`state["messages"]`) to the tool-bound LLM and increments `llm_calls`.
2. **`should_continue` Edge:** Inspects the latest AI message. If `tool_calls` are present, it routes execution to `tool_node`; otherwise, it finishes at `END`.
3. **`tool_node` Node:** Executes every requested tool call from `tools_by_name`, wraps outputs in `ToolMessage` objects, and loops back to `llm_call` so the LLM can synthesize the final response.

---


### Example Prompt & Output

**Input Prompt:**
> *"Hi, I am Suman! Check the live weather in Bharatpur, calculate (250 * 4) + 150, and extract the intent and keywords of this message."*

**Agent Execution Flow:**
1. Calls `greet_user(name="Suman")`
2. Calls `get_live_weather(city="Bharatpur")`
3. Calls `calculator(expression="(250 * 4) + 150")`
4. Calls `extract_intent_and_keywords(...)`
5. Synthesizes all four tool outputs into a final natural-language response.

---

## 🛠️ Adding a New Custom Tool

 Define a Python function  decorated with `@tool`, complete with **type hints** and a descriptive **docstring**:
   ```python
   from langchain_core.tools import tool

   @tool
   def my_custom_tool(param: str) -> str:
       """Clear description of when and how the LLM should use this tool."""
       return f"Processed: {param}"
   ```
