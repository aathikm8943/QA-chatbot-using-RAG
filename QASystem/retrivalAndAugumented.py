"""
Basically here we're going to retrieve the data from the pinecone database and augment it with the question asked by the user.

Steps:

1. based on the user query we need to retrieve the data from the pinecone database
2. Embedding the user query
3. Retrieving the data from the pinecone database based on the embedding
4. Building the prompt with the retrieved data and user query from LLM
5. Sending the prompt to the LLM and getting the response
6. Returning the response to the user
"""

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator
from haystack.utils.hf import HFGenerationAPIType
from haystack.utils import Secret
from haystack.components.generators.utils import ChatMessage
from QASystem.utils import pinecone_config
from QASystem.ingestion import create_ingest_pipeline
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

document_store = pinecone_config()

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

def prompt_to_messages(prompt: str):
    return [ChatMessage(role="user", content=prompt)]

def retrieval_pipeline(query):
    # Initializing the pipeline
    retrieval_pipeline = Pipeline()

    # Adding the components
    retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    retrieval_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
    retrieval_pipeline.add_component("prompt_builder", PromptBuilder(prompt_template))
    retrieval_pipeline.add_component("prompt_to_messages", prompt_to_messages)
    # retrieval_pipeline.add_component("generator", HuggingFaceTGIChatGenerator(
    #                                                     api_type="serverless_inference_api",
    #                                                     api_params= {'model': 'mistralai/Mistral-7B-v0.1'}
    #                                                     ))

    # Connecting the components in the pipeline
    retrieval_pipeline.connect("text_embedder", "retriever")
    retrieval_pipeline.connect("retriever", "prompt_builder")
    retrieval_pipeline.connect("prompt_builder", "prompt_to_messages")
    retrieval_pipeline.connect("prompt_to_messages", "generator")

    query = query

    results = retrieval_pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query}
    })

    return results["generator"]["text"]

if __name__ == "__main__":    
    # Ensure the document store is configured before running the ingestion pipeline
    # create_ingest_pipeline(document_store)
    
    query = "What is the main topic of the document? and First chapter name of the document?"
    response = retrieval_pipeline(query)
    print("Response:", response)