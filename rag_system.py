import os
import glob
import textwrap
from typing import List, Tuple
import requests
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = input(str('PLEASE ENTER OPENAI_API_KEY. I know it should be env variable but I had a key provided by my exisiting org, hence nedded to hardcode.'))

def fetch_github_files(repo_owner, repo_name, branch="main"):
    """
    Fetch all .md files from a GitHub repository.
    took help from stack overflow for funtion to browse and fetch files from github
    """
    base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    headers = {"Accept": "application/vnd.github.v3+json"}

    print(f"Fetching file list from GitHub: {repo_owner}/{repo_name} [{branch}]...")
    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        print("Error fetching repository files:", response.text)
        return []

    files_data = response.json()
    md_files = [file for file in files_data.get("tree", []) if file["path"].endswith(".md")]

    print(f"Found {len(md_files)} markdown files. Downloading...")
    
    downloaded_files = []
    
    for file in tqdm(md_files, desc="Downloading Files", unit="file"):
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
        file_content = requests.get(raw_url).text
        downloaded_files.append((file["path"], file_content))

    return downloaded_files

'''
def parse_markdown_files(root_dir: str) -> List[Document]:
    """Recursively load all .md files in root_dir into LangChain Document objects."""
    md_files = glob.glob(os.path.join(root_dir, '**', '*.md'), recursive=True)
    documents = []

    for file_path in md_files:
        # Extract immediate subfolder name as "course_name"
        parts = file_path.split(os.sep)
        course_name = parts[1] if len(parts) > 1 else "UnknownCourse"

        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()

        for doc in raw_docs:
            doc.metadata['course_name'] = course_name
            doc.metadata['source_file'] = file_path

        documents.extend(raw_docs)

    return documents
    '''

def parse_markdown_files_from_github(repo_owner, repo_name, branch="main"):
    """
    Fetch markdown files from GitHub and convert them into LangChain Document objects.
    og code
    """
    md_files = fetch_github_files(repo_owner, repo_name, branch)
    documents = []
    
    for file_path, content in md_files:
        # Extract the course name (first subfolder in repo structure)
        parts = file_path.split("/")
        course_name = parts[0] if len(parts) > 1 else "UnknownCourse"

        # Convert to LangChain Document format
        doc = Document(
            page_content=content,
            metadata={"course_name": course_name, "source_file": file_path}
        )
        documents.append(doc)

    return documents

def create_vector_store(
    documents: List[Document],
    persist_directory: str = 'chroma_db',
    batch_size: int = 100
):
    """
    Split documents into manageable chunks, embed them, and store them in Chroma. 
    This function persists the vector store to disk.
    partial LLM partial og
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in documents for chunk in text_splitter.split_text(doc.page_content)
    ]

    #didn't have time to create and store env variables.
    embeddings = OpenAIEmbeddings(
        openai_api_key= OPENAI_API_KEY
    )

    vectorstore = None
    
    #chunking wasn't smooth hence took help from internet
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vectorstore.add_documents(batch)

    return vectorstore


def load_vector_store(persist_directory: str = 'chroma_db'):
    """Load a persisted Chroma vector store from disk."""
    #og code
    embeddings = OpenAIEmbeddings(
        openai_api_key= OPENAI_API_KEY
    )
    
    # Load Chroma from persistent directory
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def get_top_courses(
    user_query: str,
    vectorstore,
    k: int = 3
) -> List[Tuple[str, str]]:
    """Return up to k (course_name, snippet) pairs most relevant to user_query."""
    
    # og code
    docs = vectorstore.similarity_search(user_query, k=15)
    seen_courses = []
    recommendations = []

    for doc in docs:
        course_name = doc.metadata.get('course_name', 'Unknown')
        snippet = doc.page_content
        if course_name not in seen_courses:
            seen_courses.append(course_name)
            recommendations.append((course_name, snippet))
        if len(recommendations) == k:
            break

    return recommendations


def summarize_course(
    course_name: str,
    snippet: str,
    query: str,
    llm
) -> str:
    
    # partial og partial internet
    """Generate a short summary explaining why the course is relevant and what it covers."""
    prompt_str = """
    You are a helpful assistant. A user has asked: "{query}"

    We have a course titled "{course_name}" with the following snippet of content:
    "{snippet}"

    1. Explain in 1-2 sentences why this course is a good match for the user's query.
    2. Provide 2-3 key learning outcomes for the course.
    3. Suggest a recommended order or who should take this course (beginners, intermediate, etc.).

    Return your answer in a concise format:
    - Why it matches
    - Key outcomes
    - Recommended order
    """

    prompt = PromptTemplate(
        input_variables=["course_name", "snippet", "query"],
        template=prompt_str.strip()
    )
    
    pipeline = prompt | llm
    response = pipeline.invoke({
        'course_name': course_name,
        'snippet': snippet,
        'query': query
    })

    return response.strip()


def main():
    """CLI loop to accept queries, retrieve course recommendations, and summarize them."""
    
    # almost og code.
    persist_dir = 'chroma_db'
    repo_owner = "0xmetaschool"
    repo_name = "Learning-Projects"
    branch = "main"
    
    if not os.path.exists(persist_dir):
        print("No existing vector store found, creating a new one...")
        docs = parse_markdown_files_from_github(repo_owner, repo_name, branch)
        vectorstore = create_vector_store(docs, persist_dir)
    else:
        print("Loading existing vector store...")
        vectorstore = load_vector_store(persist_dir)

    llm = OpenAI(
        temperature=0,
        openai_api_key= OPENAI_API_KEY
    )

    print("\n--- Metaschool Course Recommender CLI ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        
        #user_query = 'I want to learn solidity which course should I start from?'
        user_query = input("\nAsk: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        recommendations = get_top_courses(user_query, vectorstore, k=3)
        if not recommendations:
            print("No relevant courses found.")
            continue

        print(f"\n--- Recommendations for: \"{user_query}\" ---\n")
        for i, (course_name, snippet) in enumerate(recommendations, start=1):
            summary = summarize_course(course_name, snippet, user_query, llm)
            snippet_preview = textwrap.shorten(snippet, width=100, placeholder="...")

            print(f"* Recommendation #{i}: {course_name} *")
            print(f"Snippet Preview: {snippet_preview}\n")
            print(summary)
            print("-" * 50)


if __name__ == "__main__":
    main()