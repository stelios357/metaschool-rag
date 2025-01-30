import os
import re
from typing import List
from dotenv import load_dotenv
from collections import defaultdict

# --- LangChain Imports ---
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load API Key from .env file
load_dotenv("abc.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------- CONFIG: EMBEDDING MODEL ---------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def filter_file(file_name: str) -> bool:
    """Skip unnecessary files from indexing."""
    name_lower = file_name.lower()
    return any(keyword in name_lower for keyword in ["readme", "license", "contributing"])


def basic_clean(text: str) -> str:
    """Removes HTML tags and unnecessary lines."""
    clean_text = re.sub(r"<.*?>", "", text)  # Remove HTML
    lines = clean_text.splitlines()
    return "\n".join(line for line in lines if "ALL-CONTRIBUTORS" not in line and "badge" not in line)


def chunk_for_embedding(text: str, chunk_size=1000, overlap=100) -> List[Document]:
    """Splits text into smaller chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]


def load_course_data(repo_path: str) -> List[Document]:
    """Loads, cleans, and chunks course data from Markdown files."""
    all_docs = []
    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if not file_name.lower().endswith(".md") or filter_file(file_name):
                continue

            file_path = os.path.join(root, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                cleaned_text = basic_clean(f.read())

            docs = chunk_for_embedding(cleaned_text, chunk_size=1000, overlap=100)
            for d in docs:
                d.metadata = {"file_name": file_name, "rel_path": os.path.relpath(file_path, repo_path)}
            all_docs.extend(docs)

    return all_docs


def build_vector_store(docs: List[Document]) -> FAISS:
    """Creates a FAISS vector store using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(docs, embeddings)


def group_chunks_by_course(chunks):
    """Groups document chunks by course name."""
    course_groups = defaultdict(list)
    for chunk in chunks:
        course_id = chunk.metadata["file_name"]
        course_groups[course_id].append(chunk)
    return course_groups


def rank_courses(course_groups):
    """Ranks courses based on their chunk scores."""
    ranked = []
    for course_id, chunks in course_groups.items():
        # Consider the total score of all chunks from the same course
        total_score = sum(chunk.metadata.get("score", 0) for chunk in chunks)
        ranked.append((course_id, total_score))

    # Sort courses by total score (descending)
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [course_id for course_id, _ in ranked[:3]]  # Return top 3 course IDs


def main():
    """Main function to run the RAG-based course recommendation system."""
    repo_path = "Learning_Projects"
    print("Reading and processing Markdown files...")
    
    docs = load_course_data(repo_path)
    print(f"Loaded {len(docs)} text chunks.")

    print("Building vector store with HuggingFace embeddings...")
    vector_store = build_vector_store(docs)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True
    )

    print("\nMetaschool RAG System (Type 'exit' to quit)")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        result = qa_chain({"query": user_query})
        chunks = result.get("source_documents", [])

        if not chunks:
            print("No relevant courses found. Try refining your query.")
            continue

        course_groups = group_chunks_by_course(chunks)
        top_courses = rank_courses(course_groups)

        prompt = (
            "Based on these top 3 courses, recommend them in order, explain why they are relevant, "
            "and provide a summary:\n" + "\n".join(top_courses)
        )
        answer = llm.invoke(prompt)
        
        print("\nRecommended Courses:\n", answer)


if __name__ == "__main__":
    main()
