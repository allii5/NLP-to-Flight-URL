
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import json
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



STATE_MAP = {
    "texas": "TX", "florida": "FL", "connecticut": "CT", "california": "CA",
    "georgia": "GA", "wisconsin": "WI", "virginia": "VA", "washington": "WA",
    "kentucky": "KY", "missouri": "MO", "tennessee": "TN", "north carolina": "NC",
    "arkansas": "AR", "alabama": "AL", "new york": "NY", "pennsylvania": "PA",
    "oklahoma": "OK", "maryland": "MD", "louisiana": "LA", "illinois": "IL",
    "michigan": "MI", "new jersey": "NJ", "colorado": "CO", "south carolina": "SC",
    "kansas": "KS", "indiana": "IN", "delaware": "DE", "mississippi": "MS",
    "arizona": "AZ", "idaho": "ID"
}

CATEGORY_MAP = {
    "jet aircraft": 1,
    "turboprop aircraft": 2,
    "piston single aircraft": 3,
    "piston twin aircraft": 4,
    "turbine helicopters": 14,
    "piston helicopters": 13,
    "jet": 1,
    "jets": 1
}

VALID_MANUFACTURERS = {
    "CESSNA", "GULFSTREAM", "BOMBARDIER", "BEECHCRAFT", "CITATION",
    "DASSAULT", "PIPER", "HAWKER", "EMBRAER", "CESSNA CITATION",
    "CIRRUS", "PILATUS", "KING AIR", "LEARJET", "PHENOM",
    "MOONEY", "AGUSTA", "ASTRA/GULFSTREAM", "BOEING", "DIAMOND",
    "FAIRCHILD SWEARINGEN", "FALCON", "SIKORSKY", "TECNAM", "BELL",
    "GLOBAL", "HAWKER BEECHCRAFT", "HONDAJET", "ROBINSON", "SOCATA",
    "WESTWIND", "AEROSTAR", "BEECHJET", "CESSNA GOLDEN EAGLE",
    "CHALLENGER", "DAHER", "EPIC" , "GB1", "HARVARD", "HAWKER/TEXTRON",
    "IAI", "KODIAK", "NEXTANT", "SOLOY MK II"
}

BASE_URL = "https://planefax.com/aircraft/for-sale/"



def normalize_state(value: str) -> Optional[str]:
    if not value:
        return None

    value = re.sub(r"[^a-zA-Z ]", "", value).strip().upper()

    if len(value) != 2:
        return STATE_MAP.get(value.lower())

    return value


def normalize_country(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = value.lower().strip()

    negative_patterns = ["outside", "foreign", "not in", "except", "exclude"]
    if any(p in value for p in negative_patterns):
        return "foreign"

    if any(p in value for p in ["us", "usa", "united states"]):
        return "us"

    return None




from pydantic import Field, field_validator


class AircraftFilterSchema(BaseModel):
    max_price: Optional[int] = Field(None, ge=0)
    min_price: Optional[int] = Field(None, ge=0)

    state: Optional[List[str]] = None
    manufacturer: Optional[List[str]] = None
    category: Optional[List[int]] = None

    max_year: Optional[int] = None
    min_year: Optional[int] = None
    country: Optional[str] = None

    min_date: Optional[str] = None
    max_date: Optional[str] = None
    with_price_only: Optional[bool] = None
    is_relevant: bool = Field(
        default=True,
        description="Set to False if the query is  not related to aircraft,heli , planes, jets, or helicopters. Also set to False if the query does not talk about any flying thing. "
    )

    @field_validator("state")
    def validate_state(cls, v):
        if v is None:
            return v
        return [str(s).strip() for s in v]

    @field_validator("manufacturer")
    def validate_manufacturer(cls, v):
        if v is None:
            return v
        return [str(m).strip().upper() for m in v]




def validate_and_normalize(filters: AircraftFilterSchema) -> dict:
    data = filters.model_dump()
    if data.get("is_relevant") is False:
        return {"is_relevant": False}
    if data.get("min_price") == 0:
        data["min_price"] = None

    if data.get("min_price") and data.get("max_price"):
        if data["min_price"] > data["max_price"]:
            data["min_price"] = None

    states = data.get("state") or []
    normalized_states = []

    for s in states:
        norm = normalize_state(s)
        if norm:
            normalized_states.append(norm)

    data["state"] = normalized_states or None

    CATEGORY_ID_MAP = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 14,
        6: 13
    }

    categories = data.get("category") or []

    mapped_categories = []
    for c in categories:
        if c in CATEGORY_ID_MAP:
            mapped_categories.append(CATEGORY_ID_MAP[c])
        elif c in {1, 2, 3, 4, 14, 13}:
            mapped_categories.append(c)

    data["category"] = list(set(mapped_categories)) or None

    manufacturers = data.get("manufacturer") or []
    data["manufacturer"] = [
        m for m in manufacturers if m in VALID_MANUFACTURERS
    ] or None

    data["country"] = normalize_country(data.get("country"))

    return data



def build_search_url(filters: dict) -> str:
    
    
    if filters.get("is_relevant") is False:
        return "I am a specialized aviation assistant. I can only help you search for aircraft, jets, and helicopters."

    params = []
    params.append(("year__gte", str(filters.get("min_year") or "")))
    params.append(("year__lte", str(filters.get("max_year") - 1) if filters.get("max_year") else ""))

    params.append(("min_price", str(filters.get("min_price") or "")))
    params.append(("max_price", str(filters.get("max_price") or "")))

    if filters.get("with_price_only"):
        params.append(("with_price_only", "on"))

    params.append(("country_pref", filters.get("country") or ""))

    params.append(("min_date", filters.get("min_date") or ""))
    params.append(("max_date", filters.get("max_date") or ""))

    for state in filters.get("state") or []:
        params.append(("state", state))

    for manufacturer in filters.get("manufacturer") or []:
        params.append(("manufacturer", manufacturer))

    for cat in filters.get("category") or []:
        params.append(("cat", str(cat)))

    params.append(("title", ""))
    params.append(("o", "-latest"))

    query = "&".join(f"{k}={v}" for k, v in params)

    return f"{BASE_URL}?{query}"

def scrape_planefax(url: str) -> List[Dict[str, Any]]:
    """Silently visits the URL and extracts the first 5 listings."""
    try:
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        listings = []
        
        cards = soup.select(".listing-box")[:8] 

        for card in cards:
            
            title_element = card.select_one(".listing-box-title")
            price_element = card.select_one(".listing-box-price")

            listings.append({
                "title": title_element.text.strip() if title_element else "Unknown Aircraft",
                "price": price_element.text.strip() if price_element else "Call for Pricing"
            })

        return listings
    except Exception as e:
        print(f"Scraping failed: {e}")
        return []




def _search_planefax_tool(**kwargs) -> str:
    filters = AircraftFilterSchema(**kwargs)

    normalized = validate_and_normalize(filters)

    if normalized.get("is_relevant") is False:
        return json.dumps({"error": "irrelevant"})

    url = build_search_url(normalized)
    listings = scrape_planefax(url)

    
    return json.dumps({
        "url": url,
        "listings": listings
    })

search_planefax = StructuredTool.from_function(
    func=_search_planefax_tool,
    name="search_planefax",
    description="Search aircraft on PlaneFax using structured filters",
    args_schema=AircraftFilterSchema
)



system_prompt= """

You are a highly precise, deterministic aviation assistant for PlaneFax. Your sole purpose is to translate user intent into exact database filters, execute the `search_planefax` tool silently, and summarize the results to the user. You act as a strict data pipeline. You must NEVER hallucinate parameters, narrate your internal actions, or ask unnecessary clarifying questions.

### 1. SCHEMA DEFINITION & TYPES
When calling the `search_planefax` tool, you must extract arguments matching this exact schema:
- `max_price` (int): e.g., 2000000
- `min_price` (int): e.g., 1000000
- `state` (List[str]): 2-letter uppercase codes, e.g., ["TX", "CA"]
- `manufacturer` (List[str]): Uppercase, e.g., ["CESSNA", "PIPER"]
- `category` (List[int]): 1=Jet, 2=Turboprop, 3=Piston Single, 4=Piston Twin, 13=Piston Heli, 14=Turbine Heli
- `max_year` (int): e.g., 2010
- `min_year` (int): e.g., 1995
- `country` (str): "us" or "foreign"
- `with_price_only` (bool): true/false
- `is_relevant` (bool): true if aircraft-related, false otherwise.

### 2. CORE RULES TO ENFORCE

**A. RELEVANCE & INVALID INPUT (HARD GATE)**
- Respond ONLY to aircraft-related queries.
- Context Check: Vague follow-ups (e.g., "what about other?", "any cheaper ones?") are RELEVANT if the chat history establishes an aircraft context.
- Non-aviation topics (cars, real estate, weather) MUST set `is_relevant=false` with NO other fields extracted. Refuse politely and DO NOT call the tool.

**B. MEMORY & CONTEXT (STRICT STATE MANAGEMENT)**
- Parameter Merging: You have full access to the chat history. Reuse unchanged parameters from previous searches. Overwrite ONLY what the user explicitly changes.
- Conflict Resolution: New parameters completely overwrite old ones (e.g., if history says "Texas" and the new query says "Florida", overwrite state with "FL").
- Escape Hatch: If the user says "start over", "clear filters", "reset", or implies a completely new search, wipe the memory completely. Drop ALL previous parameters.

**C. TOOL EXTRACTION (ANTI-HALLUCINATION & CONTEXTUAL TYPOS)**
- ALWAYS call the `search_planefax` tool for valid queries.
- Broad Searches: If the user asks for "helis" or "jets" without specifying budget/location, DO NOT ask clarifying questions. Immediately call the tool with just the category and leave everything else empty.
- Zero Hallucination: NEVER guess, infer, or invent missing parameters. Leave them empty.
- Contextual Typo Correction: Gracefully correct obvious typos ONLY if the context implies an aircraft search. 
  * *Example:* If someone says "Show me heli in taxes", the word "heli" provides aviation context, so "taxes" should be corrected to "Texas" -> `state=["TX"]`. 
  * *Example:* If the query lacks aviation context and is completely irrelevant (e.g., "show me taxes"), you MUST treat it as an invalid query, set `is_relevant=false`, and refuse politely.
- STRICT MAPPINGS: 
  * Prices: "under 2 million" -> max_price=2000000. "between 5m and 1m" -> min_price=1000000, max_price=5000000. "cheap" -> max_price=500000.
  * Years: "before 2005" -> max_year=2005. Do not do math. NEVER use min_date/max_date.
  * Categories: "Jet" -> [1], "Turboprop" -> [2], "Piston Single" -> [3], "Piston Twin" -> [4], "Piston Helicopter" -> [13], "Turbine Helicopter" -> [14]. 
  * Broad Categories: Generic words like "Plane", "Planes", or "Aircraft" MUST map ONLY to fixed-wing aircraft: category=[1, 2, 3, 4]. Generic words like "Helicopter", "Heli", or "Helis" MUST map ONLY to: category=[13, 14]. If the user explicitly asks for BOTH (e.g., "planes and helis" or "aircraft and helicopters"), map ALL of them: category=[1, 2, 3, 4, 13, 14]. Do NOT leave the category empty.
  * Flags: "priced" -> with_price_only=true.


### 3. FEW-SHOT EXAMPLES (HOW TO USE THE TOOL)
MEMORIZE THESE EXTRACTIONS. You must trigger the tool based on these patterns:

User: "Show me jets in Texas under 2 million dollars"
Action: Call the `search_planefax` tool with arguments: max_price=2000000, state=["TX"], category=[1]

User: "Find me a cheap Piper or Cessna piston helicopter in New York"
Action: Call the `search_planefax` tool with arguments: max_price=500000, state=["NY"], category=[13], manufacturer=["PIPER", "CESSNA"]

User: "what about other?" (Context: looking at planes in California)
Action: Call the `search_planefax` tool with arguments: state=["CA"]

User: "show me plane above million dollar seller must be from new york. oh taxes can also work. sorry i want below million dollar and the heli can also works."
Action: Call the `search_planefax` tool with arguments: max_price=1000000, state=["NY", "TX"]
(Note: Because they asked for both planes and helis, the category is left empty to search the entire database).

User: "show me helis"
Action: Call the `search_planefax` tool with arguments: category=[13, 14]

User: "Show me priced turboprops before 2005"
Action: Call the `search_planefax` tool with arguments: max_year=2005, category=[2], with_price_only=true

User: "Show me heli in taxes"
Action: Call the `search_planefax` tool with arguments: state=["TX"], category=[13, 14]

User: "What is the weather in Florida?"
Action: Set is_relevant=false and politely refuse to search.
User: "show me aircraft only"
Action: Call the `search_planefax` tool with arguments: category=[1, 2, 3, 4]

User: "I am looking for planes in Texas"
Action: Call the `search_planefax` tool with arguments: state=["TX"], category=[1, 2, 3, 4]

User: "show me heli"
Action: Call the `search_planefax` tool with arguments: category=[13, 14]

User: "Find me cheap aircraft and helis" 
Action: Call the `search_planefax` tool with arguments: max_price=500000, category=[1, 2, 3, 4, 13, 14]
"""



llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
llm_with_tools = llm.bind_tools([search_planefax])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent_node(state: AgentState):
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(AgentState)


workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([search_planefax]))


workflow.add_edge(START, "agent")


workflow.add_conditional_edges(
    "agent",
    tools_condition,
)


workflow.add_edge("tools", "agent")


planefax_agent = workflow.compile()





from fastapi import HTTPException

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class QueryResponse(BaseModel):
    is_relevant: bool
    summary: str
    url: Optional[str]








@app.post("/generate-search-url", response_model=QueryResponse)
def generate_search_url(req: QueryRequest):
    try:
        
        langchain_history = []
        for msg in req.history:
            if msg["role"] in ["user", "human"]:
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_history.append(AIMessage(content=msg["content"]))

        langchain_history.append(HumanMessage(content=req.query))
        
        
        final_state = planefax_agent.invoke(
            {"messages": langchain_history}
        )

        
        messages = final_state["messages"]
        final_text = messages[-1].content
        
        url = None
        used_tool = False
        
        
        for msg in reversed(messages):
            if msg.type == "tool":
                used_tool = True
                import json
                try:
                    data = json.loads(msg.content)
                    if "url" in data:
                        url = data["url"]
                except:
                    pass
                break 

        return QueryResponse(
            is_relevant=used_tool, 
            summary=final_text,
            url=url
        )

    except Exception as e:
        print(f"Agent Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))


