#Names: Sameer Shaik and Mohammed Irfan Battegeri
#Assignment: Final Project (Pdf Query Answer System)
#Course: CSC 575
#Section: 801_1125



from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os


#Create required dirs
def create_dirs(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Created directory {file_path}")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the directory where you want to store the file
DATA_PATH = os.path.join(script_dir, 'data/')
create_dirs(DATA_PATH)
DB_FAISS_PATH = os.path.join(script_dir, 'VectorStore/')
create_dirs(DB_FAISS_PATH)

#Deleting a file from DATA_PATH
def safe_delete_file(file_path):
    #Safely delete a file, ensuring it exists before attempting deletion.
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Successfully deleted {file_path}")
    else:
        print(f"File {file_path} does not exist")

# Adjust the text splitter to create smaller chunks
def create_vector_db(uploaded_file):
    temp_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Now, assume you need a file path to load the document
    loader = PyPDFLoader(file_path=temp_path)
    documents = loader.load() 

    # Continue with the processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError("No text chunks were extracted from the PDF.")
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)    

    # After successful embedding and saving, delete the PDF file
    safe_delete_file(temp_path)

llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",   # We can use different models too...
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


# Adjust QA retrieval to handle token limit
def retrieval_qa_chain(llm, prompt, db):
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Adjusted to a valid chain type
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Define the prompt template
def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't find relevant information in the documents, clearly state that the question is out of scope.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:

    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def generate_query_embedding(query, model_name):
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Generate and return the embedding
    return model.encode(query, convert_to_tensor=False)  # No need to convert to tensor here

def final_result(query):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generate the query embedding
    query_embedding = generate_query_embedding(query, model_name)

    # Load the FAISS index
    index = faiss.read_index(DB_FAISS_PATH + 'index.faiss')

    # Perform the search in the FAISS index
    D, I = index.search(np.array([query_embedding]).astype('float32'), 10)  # Search for the top 10 similar embeddings
    
    # Print out similarity scores and indexes of the similar embeddings
    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        print(f"Rank {i+1}: Index {idx} with distance {distance}")

    # Proceed with the rest of the QA process
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    qa_result = qa({'query': query})
    return qa_result['result']