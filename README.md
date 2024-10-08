# PDF Query Answer System using RAG

## Overview
This repository contains the code for our PDF Query Answer System, developed as part of our final project for CSC 575. The system uses **Llama 2**, **RAG (Retrieval Augmented Generation)**, **Faiss**, **Langchain**, and **Streamlit** to process PDF documents, extract relevant information, and provide concise answers to user queries. This project offers a highly interactive and efficient way to search and retrieve information from PDFs, making it a powerful tool for both educational and professional environments.

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Further Enhancements](#further-enhancements)
- [Contributors](#contributors)

## Introduction
In this project, we built a PDF query answer system that leverages advanced natural language processing (NLP) techniques to enable users to upload PDFs and ask queries. The system processes the PDF, retrieves relevant sections, and responds with concise, accurate answers.

The project combines powerful machine learning techniques such as **Llama 2** for language understanding and response generation, **Faiss** for efficient similarity searches, **Langchain** for managing NLP workflows, and **Streamlit** for creating an interactive user interface.

## System Architecture
Our system integrates multiple components for efficient retrieval and query answering:
- **Llama 2** is used to understand natural language queries and generate appropriate responses.
- **Faiss** performs similarity searches on the PDF's text embeddings to retrieve relevant sections quickly.
- **Langchain** orchestrates the interaction between the user query, Llama 2, and the retrieval system.
- **Streamlit** provides a user-friendly interface to interact with the system.

### High-Level Workflow
1. Upload PDF documents via the Streamlit UI.
2. Extract text from the PDF using **PyPDFLoader**.
3. Generate semantic embeddings using **SentenceTransformers**.
4. Use **Faiss** to index the embeddings and retrieve the most relevant sections in response to a user query.
5. Pass the query and retrieved data to **Llama 2** for generating answers.
6. Display the answer to the user in the UI.

## Key Features
- **PDF Upload and Query:** Upload multiple PDFs and ask queries to get specific answers from the documents.
- **Fast and Efficient Retrieval:** Uses Faiss indexing to quickly find relevant sections.
- **Interactive UI:** Built with Streamlit for a seamless user experience.
- **Natural Language Understanding:** Llama 2 provides accurate and contextually appropriate answers.
- **Session Management:** Users can maintain context across multiple queries.

## Technologies Used
- [**Llama 2**](https://github.com/facebookresearch/llama): Language model for natural language understanding.
- [**RAG (Retrieval Augmented Generation)**](https://huggingface.co/transformers/model_doc/rag.html): Retrieval-augmented generation model.
- [**Faiss**](https://github.com/facebookresearch/faiss): Library for efficient similarity search.
- [**Langchain**](https://github.com/hwchase17/langchain): Framework for developing applications with LLMs.
- [**Streamlit**](https://streamlit.io/): Web framework for building interactive UIs.
- [**SentenceTransformers**](https://www.sbert.net/): Library for generating text embeddings.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Install the following Python libraries:
    ```bash
    pip install streamlit sentence-transformers faiss-cpu langchain pypdf
    ```

### Running the Project Locally
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pdf-query-answer-system.git
    ```
2. Navigate to the project directory:
    ```bash
    cd pdf-query-answer-system
    ```
3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. Upload PDF files and input your query to see the results!

## How to Use
- **Step 1:** Open the Streamlit app.
- **Step 2:** Upload one or more PDF files.
- **Step 3:** Type your query into the input box and press Enter.
- **Step 4:** The system retrieves and displays the most relevant answer.
