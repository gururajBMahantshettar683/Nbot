# retrieve_with_groq.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GROQ_API_KEY = "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"  # Safer key loading
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated model name

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)
faiss_db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Step 1: Setup LLM
llm_model = ChatGroq(
    model_name=GROQ_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)

# Step 2: Retrieve docs
def retrieve_docs(query, user=None):
    extra_keywords = []
    if user:
        # Add keywords from deficiency summary
        deficiency_summary = user.get('deficiency_summary', '')
        if deficiency_summary:
            # Extract phrases like 'iron deficiency', 'B12 deficiency'
            for kw in deficiency_summary.split(','):
                if 'deficiency' in kw.lower():
                    extra_keywords.append(kw.strip())
        # Add recent symptoms
        for s in user.get('recent_symptoms', []):
            if s.get('symptom'):
                extra_keywords.append(str(s['symptom']))
        # Add diet type if present
        if user.get('diet_type'):
            extra_keywords.append(user['diet_type'])
    # Combine query and extra keywords
    full_query = query
    if extra_keywords:
        full_query += ' ' + ' '.join(extra_keywords)
    return faiss_db.similarity_search(full_query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Step 3: Structured prompt
custom_prompt_template = """
You are a dedicated and knowledgeable nutrition assistant.

Your primary purpose is to help users understand and improve their nutritional health based on their symptoms, food intake, and lifestyle. Provide personalized, practical advice on what nutrients might be lacking and how to improve them through diet and habits.

You must follow these strict guidelines:

1. ONLY respond to questions related to:
   - Nutrition
   - Diets and deficiencies
   - Food habits
   - Symptoms and wellness
   - Vitamins and minerals
   - General health advice related to food and nutrition

2. If the question is unrelated (e.g., sports, politics, geography), respond with:
   "I'm here to help you understand your nutritional health. Please ask something related to food, symptoms, or wellness."

3. If you don’t have enough relevant information to answer confidently, respond with:
   "I'm sorry, I couldn’t find enough information in the documents to answer that."

4. Your tone should be:
   - Empathetic
   - Clear and concise
   - Conversational (use “you”)
   - Free from markdown, bullet marks (*, #), or technical formatting

5. Do NOT show any thought process, logic steps, or inner reasoning.
6. Do NOT generate answers to off-topic queries.

Use the following information to help guide your response:

User Profile:
{profile}

Conversation History:
{context}

User Question:
{question}

Answer:
"""






# Step 4: Query function

def is_greeting_or_ambiguous(query):
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    q = query.strip().lower()
    # Greeting or very short/ambiguous
    if any(g in q for g in greetings) and len(q.split()) <= 3:
        return True
    if len(q) <= 10:
        return True
    return False

def answer_query(query, user=None, conversation_context=None):
    # Use only the current query for retrieval to avoid irrelevant context bleed
    retrieval_query = query

    # For LLM context, keep the formatted conversation
    conversation_str = ""
    if conversation_context:
        for turn in conversation_context:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                conversation_str += f"User: {content}\n"
            else:
                conversation_str += f"Assistant: {content}\n"

    # Use only the current query for retrieval
    retrieved_docs = retrieve_docs(retrieval_query)
    if not retrieved_docs:
        return "I'm sorry, I couldn’t find enough information in the documents to answer that.", []
    print("\n[Retrieved Documents]")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\nDoc {i}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
    context = get_context(retrieved_docs)

    # Build user profile
    profile = ""
    if user:
        name = user.get("name", "User")
        gender = user.get("gender", "")
        dob = user.get("dob", "")
        profile += f"The user's name is {name}. "
        if gender:
            profile += f"The user identifies as {gender}. "
        if dob:
            profile += f"The user's date of birth is {dob}. "

        # Recent food logs
        recent_foods = user.get("recent_foods", [])
        food_list = [str(f) for f in recent_foods if isinstance(f, (str, dict))]
        if food_list:
            profile += f"The user recently ate: {', '.join(food_list)}. "

        # Recent symptoms
        symptoms = user.get("recent_symptoms", [])
        if symptoms:
            profile += "The user recently reported the following symptoms: "
            for s in symptoms:
                desc = s.get("description", "")
                profile += f"{s.get('symptom')} (Severity: {s.get('severity')}, Description: {desc}). "

    # Full context includes conversation
    full_context = f"{conversation_str}\n{context}"

    # Construct prompt and invoke model
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    result = chain.invoke({
        "question": query,
        "context": full_context,
        "profile": profile
    })

    return result, retrieved_docs




# CLI interface
if __name__ == "__main__":
    while True:
        query = input("\nAsk your nutrition question (or type 'quit'): ")
        if query.lower() == 'quit':
            break
        answer, docs = answer_query(query)
        print(f"\nAnswer:\n{answer}\n")
        print("Sources:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}")
