import config
from clients import get_blob_service


def ensure_container():
    container = get_blob_service().get_container_client(config.CONTAINER)
    if not container.exists():
        container.create_container()
    return container


def upload_cv(container, filename: str, data: bytes) -> None:
    container.upload_blob(filename, data, overwrite=True)