
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load a local embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create a local vector database
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.get_or_create_collection(
    name="university_faqs",
    metadata={"hnsw:space": "cosine"}
)

# 3. Prepare 10 university FAQs
faqs = [
    {
        "question": "How can I apply for university admission?",
        "answer": "Complete the online application form and submit the required documents.",
        "category": "Admissions"
    },
    {
        "question": "What documents are required for admission?",
        "answer": "Students may need academic transcripts, identification, photographs, and an application form.",
        "category": "Admissions"
    },
    {
        "question": "How can I pay my semester fee?",
        "answer": "Follow the university's official fee payment instructions to pay your semester dues.",
        "category": "Fees"
    },
    {
        "question": "Are scholarships available for students?",
        "answer": "Eligible students may apply for scholarships according to the university's financial aid policy.",
        "category": "Scholarships"
    },
    {
        "question": "Where can I find my class timetable?",
        "answer": "Check the student portal or official academic announcements for your class schedule.",
        "category": "Academics"
    },
    {
        "question": "How can I access the university library?",
        "answer": "Students can use library services according to university library rules and access procedures.",
        "category": "Library"
    },
    {
        "question": "What should I do if I forget my student portal password?",
        "answer": "Use the password reset option or contact the university's IT support team.",
        "category": "IT Support"
    },
    {
        "question": "How do I contact my academic department?",
        "answer": "Contact your department office through its official university contact details.",
        "category": "Departments"
    },
    {
        "question": "When do university examinations take place?",
        "answer": "Examination dates are announced through the official academic calendar or university notices.",
        "category": "Examinations"
    },
    {
        "question": "How can I get my student ID card?",
        "answer": "Follow the university's student ID issuance process and provide the required information.",
        "category": "Student Services"
    }
]

# 4. Add records only if the collection is empty
if collection.count() == 0:
    documents = [faq["question"] + " " + faq["answer"] for faq in faqs]
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=[f"faq_{i+1}" for i in range(len(faqs))],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "title": faq["question"],
                "category": faq["category"],
                "source": "Sample University FAQ"
            }
            for faq in faqs
        ]
    )

print(f"\nTotal FAQ records: {collection.count()}")
print("Semantic Search System is ready!")

# 5. Search using natural-language questions
while True:
    query = input("\nEnter your question (or type 'exit'): ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        print("Please enter a question.")
        continue

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nMost relevant results:")

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i+1}")
        print(f"Category: {metadata['category']}")
        print(f"Title: {metadata['title']}")
        print(f"Information: {document}")
        print(f"Cosine distance: {distance:.4f}")