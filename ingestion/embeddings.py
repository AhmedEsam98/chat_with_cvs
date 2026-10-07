import config
from clients import get_openai_client


def embed(texts: list[str], client=None) -> list[list[float]]:
    client = client or get_openai_client()
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i:i + config.EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors