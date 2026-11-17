
import re

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from reasoner import reason_about_focus
from query_planner import plan_query


# Configuration
DB_PATH = "./chroma_db"
COLLECTION = "amis_glosses"
EMBEDDING_MODEL = "bge-m3"
RESULTS_PER_QUERY = 10
MAX_EXAMPLES_PER_CATEGORY = 3
MAX_EXAMPLES = 5

FOCUS_CATEGORIES = {
    "Agent Focus": ["AF"],
    "Patient Focus": ["PF"],
    "Referential / Instrumental Focus": ["RF", "IF"],
    "Locale Focus": ["LF"]
}
# Initialize ChromaDB
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

db = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def has_marker(gloss, marker):
    """
    Matches:
        AF-finish
        AF=...
        <AF>eat

    Does not match PF inside PFV.
    """

    marker = re.escape(marker.upper())

    pattern = (
        rf"(?<![A-Za-z])"
        rf"(?:<{marker}>|{marker})"
        rf"(?=$|[-=\s])"
    )

    return bool(
        re.search(pattern, gloss, re.IGNORECASE)
    )


def retrieve_inventory(plan):

    markers = plan.get("gloss_markers", [])

    # Determine which known focus categories are relevant.
    categories = {
        category: category_markers
        for category, category_markers in FOCUS_CATEGORIES.items()
        if any(marker in markers for marker in category_markers)
    }

    inventory = {
        category: {
            "count": 0,
            "examples": []
        }
        for category in categories
    }

    total = db._collection.count()

    for offset in range(0, total, 500):

        batch = db._collection.get(
            limit=500,
            offset=offset,
            include=["documents", "metadatas"]
        )

        for gloss, metadata in zip(
            batch["documents"],
            batch["metadatas"]
        ):

            if not gloss:
                continue

            for category, category_markers in categories.items():

                if any(
                    has_marker(gloss, marker)
                    for marker in category_markers
                ):

                    inventory[category]["count"] += 1

                    if len(inventory[category]["examples"]) < MAX_EXAMPLES:

                        inventory[category]["examples"].append({
                            "sentence_id": metadata.get("sentence_id"),
                            "text": metadata.get("text", ""),
                            "gloss": gloss
                        })

    return {
        "evidence": inventory,
        "statistics": {
            category: data["count"]
            for category, data in inventory.items()
        },
        "retrieval_method": "exact_marker_scan"
    }


def retrieve_specific(plan, k=15):

    queries = plan.get("search_queries", [])
    markers = plan.get("gloss_markers", [])

    results_by_id = {}

    for query in queries:

        results = db.similarity_search_with_score(
            query,
            k=k
        )

        for doc, distance in results:

            sid = doc.metadata.get("sentence_id")

            if sid is None:
                continue

            if sid not in results_by_id or distance < results_by_id[sid]["distance"]:

                results_by_id[sid] = {
                    "sentence_id": sid,
                    "text": doc.metadata.get("text", ""),
                    "gloss": doc.page_content,
                    "distance": float(distance)
                }

    evidence = sorted(
        results_by_id.values(),
        key=lambda x: x["distance"]
    )

    return {
        "evidence": evidence[:k],
        "statistics": {},
        "retrieval_method": "semantic_search",
        "markers_searched": markers
    }

def retrieve_comparison(plan, k=5):

    categories = plan.get("categories", [])
    markers = plan.get("gloss_markers", [])

    evidence = {}

    for category in categories:

        category_markers = FOCUS_CATEGORIES.get(category, [])

        if not category_markers:
            continue

        examples = []

        total = db._collection.count()

        for offset in range(0, total, 500):

            batch = db._collection.get(
                limit=500,
                offset=offset,
                include=["documents", "metadatas"]
            )

            for gloss, metadata in zip(
                batch["documents"],
                batch["metadatas"]
            ):

                if not gloss:
                    continue

                if any(
                    has_marker(gloss, marker)
                    for marker in category_markers
                ):

                    examples.append({
                        "sentence_id": metadata.get("sentence_id"),
                        "text": metadata.get("text", ""),
                        "gloss": gloss,
                        "category": category
                    })

        evidence[category] = {
            "count": len(examples),
            "examples": examples[:k]
        }

    return {
        "evidence": evidence,
        "statistics": {
            category: data["count"]
            for category, data in evidence.items()
        },
        "retrieval_method": "exact_marker_scan"
    }


def marker_pattern(marker):
    """
    Match a gloss marker as a standalone item or as a prefix,
    e.g. AF, AF-finish, or AF=...
    Avoid matching AF inside PFV or other longer abbreviations.
    """
    marker = re.escape(marker.upper())
    return re.compile(
        rf"(?<![A-Za-z]){marker}(?=$|[-=\s])",
        re.IGNORECASE
    )

def retrieve_focus_inventory():
    stored = db.get(include=["documents", "metadatas"])

    inventory = {}

    for category, markers in FOCUS_CATEGORIES.items():

        matching = []

        for gloss, metadata in zip(
            stored["documents"],
            stored["metadatas"]
        ):
            if not gloss or not metadata:
                continue

            if any(has_marker(gloss, marker) for marker in markers):
                matching.append({
                    "id": (
                        metadata.get("sentence_id")
                        or metadata.get("id")
                    ),
                    "text": metadata.get("text", ""),
                    "gloss": gloss
                })

        inventory[category] = {
            "markers": markers,
            "count": len(matching),
            "examples": matching[:MAX_EXAMPLES_PER_CATEGORY]
        }

    return inventory

def retrieve(question, k=RESULTS_PER_QUERY, max_marker_results=20):
    # 1. Generate retrieval plan
    plan = plan_query(question)

    print("\nRETRIEVAL PLAN")
    print(plan)

    results = {}

    # 2. Semantic vector searches
    for query in plan["search_queries"]:
        matches = db.similarity_search_with_score(query, k=k)

        for doc, distance in matches:
            sentence_id = (
                doc.metadata.get("sentence_id")
                or doc.metadata.get("id")
            )

            if sentence_id is None:
                continue

            sentence_id = str(sentence_id)

            if sentence_id not in results:
                results[sentence_id] = {
                    "document": doc,
                    "distance": distance,
                    "match_type": "vector"
                }
            else:
                results[sentence_id]["distance"] = min(
                    results[sentence_id]["distance"],
                    distance
                )

    # 3. Exact gloss-marker search
    if plan["gloss_markers"]:
        stored = db.get(include=["documents", "metadatas"])

        patterns = [
            marker_pattern(marker)
            for marker in plan["gloss_markers"]
        ]

        marker_count = 0

        for gloss, metadata in zip(
            stored["documents"],
            stored["metadatas"]
        ):
            if not gloss or not metadata:
                continue

            if not any(pattern.search(gloss) for pattern in patterns):
                continue

            sentence_id = (
                metadata.get("sentence_id")
                or metadata.get("id")
            )

            if sentence_id is None:
                continue

            sentence_id = str(sentence_id)

            if sentence_id not in results:
                from langchain_core.documents import Document

                doc = Document(
                    page_content=gloss,
                    metadata=metadata
                )

                results[sentence_id] = {
                    "document": doc,
                    "distance": None,
                    "match_type": "gloss marker"
                }

                marker_count += 1

                if marker_count >= max_marker_results:
                    break

            else:
                if "gloss marker" not in results[sentence_id]["match_type"]:
                    results[sentence_id]["match_type"] += " + gloss marker"

    # 4. Sort results
    sorted_results = sorted(
        results.values(),
        key=lambda item: (
            item["distance"] is None,
            item["distance"] if item["distance"] is not None else 0
        )
    )

    return plan, sorted_results


def display_results(plan, results):
    print("\n" + "=" * 80)
    print(f"RETRIEVED {len(results)} SENTENCES")
    print("=" * 80)
    print(results)
    for item in results:
        doc = item["document"]
        metadata = doc.metadata

        sentence_id = (
            metadata.get("sentence_id")
            or metadata.get("id")
        )

        print(f"\nID: {sentence_id}")
        print(f"MATCH: {item['match_type']}")

        if item["distance"] is not None:
            print(f"DISTANCE: {item['distance']:.4f}")

        print(f"TEXT: {metadata.get('text', '')}")
        print(f"GLOSS: {doc.page_content}")


if __name__ == "__main__":
    result = retrieve("How does past tense works in Amis?")

    plan = result[0]
    documents = result[1]

    print("Question type:", plan["question_type"])
    print("Search queries:", plan["search_queries"])
    print("Gloss markers:", plan["gloss_markers"])
    # print("Inventory target:", plan["inventory_target"])

    for item in documents:
        doc = item["document"]

        print("\nDistance:", item["distance"])
        print("Match type:", item["match_type"])
        print("Sentence ID:", doc.metadata.get("sentence_id"))
        print("Text:", doc.metadata.get("text"))
        print("Gloss:", doc.page_content)
    # while True:
    #     question = input("\nQuestion (or 'exit'): ").strip()

    #     if question.lower() == "exit":
    #         break

    #     if not question:
    #         continue

    #     plan = plan_query(question)

    #     if plan.get("question_type") == "inventory":
    #         inventory = retrieve_focus_inventory()

    #         print("\nFOCUS INVENTORY")

    #         for category, data in inventory.items():
    #             print(
    #                 f"\n{category}: {data['count']} sentences"
    #             )

    #         answer = reason_about_focus(question, inventory)

    #         print("\nLINGUISTIC ANALYSIS")
    #         print(answer)

    #     else:
    #         # Keep your existing specific-query retrieval here.
    #         results = retrieve(question)
    #         display_results(plan, results)