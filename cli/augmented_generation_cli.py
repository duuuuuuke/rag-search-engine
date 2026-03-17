import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize the content of retrieved documents",
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top search results to include in the summary (default: 5)",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate a list of citations for the retrieved documents"
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for generating citations"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top search results to include in the citations (default: 5)",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question based on retrieved documents"
    )
    question_parser.add_argument(
        "query", type=str, help="Search query for answering the question"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top search results to include in the answer (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Answer:")
            print(result["citations"])
        case "question":
            result = question_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("Answer:")
            print(result["question_answers"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
