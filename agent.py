from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from rag import load_vector_db
from tools import parse_shipping_input, inventory_lookup, keep_inventory, parse_model_output

def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    db = load_vector_db()
    retriever = db.as_retriever()

    def rag_search(query):
        docs = retriever.get_relevant_documents(query)
        return "\n".join([d.page_content for d in docs])

    tools = [
        Tool(
            name="Medical Knowledge Base",
            func=rag_search,
            description="Use for medical logistics rules and info"
        ),
        Tool(
            name="Shipping Calculator",
            func=lambda x: str(parse_shipping_input(x)),
            description="Input format: distance_km,weight_kg"
        ),
        Tool(
            name="Inventory Lookup",
            func=inventory_lookup,
            description="Input format: item_name,required_amount. Returns availability and keep-inventory instructions."
        ),
        Tool(
            name="Keep Inventory",
            func=keep_inventory,
            description="Input format: item_name,amount,yes. Confirms inventory is kept unchanged."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent


def run_agent_with_parsing(user_input):
    agent = get_agent()
    raw = agent.run(user_input)
    parsed = parse_model_output(raw)
    return {
        "raw": raw,
        "parsed": parsed
    }