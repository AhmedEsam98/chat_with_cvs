import streamlit as st
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient

import config


@st.cache_resource
def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=config.AOAI_ENDPOINT,
        api_key=config.AOAI_API_KEY,
        api_version=config.AOAI_API_VERSION,
    )


@st.cache_resource
def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        config.SEARCH_ENDPOINT, AzureKeyCredential(config.SEARCH_KEY))


@st.cache_resource
def get_search_client() -> SearchClient:
    return SearchClient(
        config.SEARCH_ENDPOINT,
        config.INDEX_NAME,
        AzureKeyCredential(config.SEARCH_KEY),
    )


@st.cache_resource
def get_blob_service() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(
        config.STORAGE_CONNECTION_STRING)
