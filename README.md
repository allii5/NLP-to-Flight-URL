# ✈️ AeroQuery AI: NLP-to-URL Aviation Assistant

A highly precise, LangGraph-powered AI backend designed to translate complex natural language queries into exact database filters and search URLs for the PlaneFax aviation marketplace.

Unlike standard conversational chatbots, this system acts as a **strict, deterministic data pipeline**. It uses an Agentic State Machine to parse user intent, map it against hardcoded dictionaries, and dynamically construct complex URL parameters—all while maintaining a conversational frontend experience.

## 🚀 Key Features

* **LangGraph State Machine:** Replaces fragile prompt-based memory with explicit state management (`TypedDict`) and deterministic node routing, ensuring perfect parameter merging across multi-turn conversations.
* **Zero-Hallucination Tool Calling:** Enforces strict Pydantic schema validation (`AircraftFilterSchema`). If the LLM generates an invalid parameter, the system traps it before it hits the database.
* **Negative Constraint Handling:** Capable of resolving complex exclusions (e.g., *"Show me planes in all states except Texas"*) by dynamically calculating and injecting the inverse arrays.
* **Contextual Typo Correction:** Intelligently corrects domain-specific typos based on aviation context (e.g., mapping "taxes" to "Texas" only when discussing aircraft).
* **Silent Execution & Web Scraping:** Automatically visits the generated URL, scrapes the top marketplace listings via `BeautifulSoup`, and returns a clean, bulleted summary to the user.

## 🏗️ Architecture

This project is decoupled into a robust API backend that can plug into any frontend (like Streamlit or React).

1.  **FastAPI Bridge:** Receives the user query and chat history.
2.  **LangGraph Agent Node:** Ingests the history, applies the master system prompt, and outputs JSON arguments.
3.  **Conditional Routing:** Forces the flow to the `ToolNode` if JSON is generated, preventing conversational leakage.
4.  **Python Execution Node:** Validates the JSON via Pydantic, builds the PlaneFax URL, executes the HTTP request, and returns strict double-quoted JSON strings back to the LLM.

## 🛠️ Tech Stack

* **Framework:** FastAPI
* **AI/Orchestration:** LangGraph, LangChain (`init_chat_model`)
* **LLM:** OpenAI (`gpt-4o-mini`)
* **Data Validation:** Pydantic
* **Web Scraping:** Requests, BeautifulSoup4

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/aeroquery-ai.git](https://github.com/yourusername/aeroquery-ai.git)
   cd aeroquery-ai