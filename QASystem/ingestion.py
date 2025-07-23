"""
1. Convert the pdf to documents(text)
2. converting the documents to chunks
3. Chunking to embeddings
4. Indexing the embeddings to the document store

"""

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.writers import DocumentWriter

from pathlib import Path
import os
from dotenv import load_dotenv
from QASystem.utils import pinecone_config

def create_ingest_pipeline(document_store):

    ## Initializing the pipeline
    indexing_pipeline = Pipeline()

    ## Adding the components
    indexing_pipeline.add_component("pdf_to_document", PyPDFToDocument())
    indexing_pipeline.add_component("document_splitter", DocumentSplitter(split_length=2, split_by="sentence"))
    indexing_pipeline.add_component("document_embedder", SentenceTransformersDocumentEmbedder())
    indexing_pipeline.add_component("document_writer", DocumentWriter(document_store=document_store))

    ## Connecting the components in the pipeline
    indexing_pipeline.connect("pdf_to_document", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    ## Run the pipeline with the local document
    indexing_pipeline.run({
        "pdf_to_document": {"sources": [Path(r"data\\Real World Research PDF.pdf")]}})

if __name__ == "__main__":
    document_store = pinecone_config()
    create_ingest_pipeline(document_store)
    print("Ingestion pipeline created successfully.")