from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    LexicalAnalyzerName,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

import config
from clients import get_index_client, get_search_client


def get_strengthened_index_definition() -> SearchIndex:
    """Define the production-grade, strengthened schema for Azure AI Search."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="cv_name", type=SearchFieldDataType.String,
                        filterable=True, facetable=True),
        SearchableField(name="candidate_name", type=SearchFieldDataType.String,
                        filterable=True, facetable=True, sortable=True),
        SearchableField(name="section_name", type=SearchFieldDataType.String,
                        filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=config.EMBED_DIM,
            vector_search_profile_name="hnsw-profile",
        ),
        SearchField(name="skills",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True, facetable=True, searchable=True),
        SearchField(name="job_titles",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True, facetable=True, searchable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="uploaded_at", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="cv-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="candidate_name"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[
                        SemanticField(field_name="cv_name"),
                        SemanticField(field_name="section_name"),
                        SemanticField(field_name="job_titles"),
                        SemanticField(field_name="skills"),
                    ],
                ),
            )
        ]
    )

    return SearchIndex(
        name=config.INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_index() -> None:
    index_client = get_index_client()
    index_def = get_strengthened_index_definition()
    index_client.create_or_update_index(index_def)


def delete_cv_chunks(cv_name: str, search_client=None) -> None:
    """Remove old chunks of a CV so re-uploads don't leave stale data."""
    client = search_client or get_search_client()
    safe = cv_name.replace("'", "''")
    old = client.search(search_text="*", filter=f"cv_name eq '{safe}'",
                        select=["id"], top=1000)
    ids = [{"id": r["id"]} for r in old]
    if ids:
        client.delete_documents(ids)


def upload_chunks(docs: list[dict], search_client=None) -> None:
    client = search_client or get_search_client()
    client.merge_or_upload_documents(docs)