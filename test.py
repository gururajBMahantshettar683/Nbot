from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory="vectorstore/chroma", embedding_function=embedding)

results = db.similarity_search("name fifferent sports")
for doc in results:
    print(doc.page_content[:300])
