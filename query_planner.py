
import json
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    format="json",
    num_predict=300,
    reasoning=False
)

CLASSIFICATION_PROMPT = """
You are a query classifier for a linguistic corpus of Amis.

Classify the user's question into exactly one category:

- inventory: asks for all categories, how many categories exist,
  or a complete inventory.
- comparison: asks to compare two or more categories or constructions.
- specific: asks about one particular linguistic phenomenon.

Return only JSON:
{"question_type": "inventory"}

Do not answer the question.
"""


INVENTORY_PROMPT = """
You are a retrieval planner for a linguistic corpus of Amis.

The user asks an inventory question.

Identify what linguistic categories should be inventoried.

Known Amis focus system:
- Agent Focus: AF
- Patient Focus: PF
- Referential / Instrumental Focus: RF, IF
- Locale Focus: LF

RF and IF are alternative labels for one category when counting.

If the question concerns the focus system, include all five markers.
Do not assume every inventory question concerns focus.

Return only JSON:
{
    "search_queries": ["query 1"],
    "gloss_markers": ["AF", "PF", "RF", "IF", "LF"],
    "inventory_target": "Amis focus system"
}

Use an empty marker list if no markers are known.
"""


COMPARISON_PROMPT = """
You are a retrieval planner for a linguistic corpus of Amis.

Identify the linguistic categories or constructions being compared.

Known Amis focus system:
- Agent Focus: AF
- Patient Focus: PF
- Referential / Instrumental Focus: RF, IF
- Locale Focus: LF

Do not confuse PF with PFV.

Return only JSON:
{
    "search_queries": ["query 1", "query 2"],
    "gloss_markers": ["AF", "PF"],
    "categories": ["Agent Focus", "Patient Focus"]
}

Only include categories and markers relevant to the question.
Do not answer the question.
"""


SPECIFIC_PROMPT = """
You are a retrieval planner for a linguistic corpus of Amis.

Identify search terms and known gloss markers relevant to the question.

Known Amis focus system:
- Agent Focus: AF
- Patient Focus: PF
- Referential / Instrumental Focus: RF, IF
- Locale Focus: LF

Do not confuse PF with PFV.
Do not invent gloss markers.
If the question explicitly mentions a marker, preserve it.

Return only JSON:
{
    "search_queries": ["query 1", "query 2"],
    "gloss_markers": ["MARKER"]
}

Do not answer the question.
"""


def plan_query(question):

    classification = llm.invoke([
        ("system", CLASSIFICATION_PROMPT),
        ("human", question)
    ])

    question_type = json.loads(
        classification.content
    )["question_type"]

    prompts = {
        "inventory": INVENTORY_PROMPT,
        "comparison": COMPARISON_PROMPT,
        "specific": SPECIFIC_PROMPT
    }


    response = llm.invoke([
        ("system", prompts[question_type]),
        ("human", question)
    ])

    plan = json.loads(response.content)

    plan["question_type"] = question_type

    return plan



if __name__ == "__main__":
    while True:
        question = input("\nQuestion (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        plan = plan_query(question)

        print("\nRETRIEVAL PLAN")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
