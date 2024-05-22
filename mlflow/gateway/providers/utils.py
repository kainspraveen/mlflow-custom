from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict
import json
import aiohttp

from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.utils.uri import append_to_uri_path


@asynccontextmanager
async def _aiohttp_post(headers: Dict[str, str], base_url: str, path: str, payload: Dict[str, Any]):
    async with aiohttp.ClientSession(headers=headers) as session:
        url = append_to_uri_path(base_url, path)
        timeout = aiohttp.ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS)
        async with session.post(url, json=payload, timeout=timeout) as response:
            print(response)
            yield response

def _aiohttp_post_bison(base_url: str, path: str, payload: Dict[str, Any]):
    import vertexai
    from vertexai.language_models import TextGenerationModel

    vertexai.init(project="db-dev-z23y-ai-survey-lab", location="europe-west1")
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1
    }
    print(payload)
    model = TextGenerationModel.from_pretrained("text-bison-32k")
    response = model.predict(
        payload.get("prompt").get("text"),
        **parameters
    )
    print(f"Response from Model: {response.text}")
    return response

def _aiohttp_post_gecko(base_url: str, path: str, payload: Dict[str, Any]):
    from vertexai.preview.language_models import TextEmbeddingModel
    import vertexai
    vertexai.init(project="db-dev-z23y-ai-survey-lab", location="europe-west1")
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual")
    print(payload)
    embeddings = model.get_embeddings(payload.get("texts"))
    embed_list = []
    for embedding in embeddings:
        vector = embedding.values
        embed_list.append(
            {"value":vector}
        )
    return {"embeddings": embed_list}


async def send_request(headers: Dict[str, str], base_url: str, path: str, payload: Dict[str, Any]):
    """
    Send an HTTP request to a specific URL path with given headers and payload.

    Args:
        headers: The headers to include in the request.
        base_url: The base URL where the request will be sent.
        path: The specific path of the URL to which the request will be sent.
        payload: The payload (or data) to be included in the request.

    Returns:
        The server's response as a JSON object.

    Raises:
        HTTPException if the HTTP request fails.
    """
    from fastapi import HTTPException

    # async with _aiohttp_post(headers, base_url, path, payload) as response:
    #     content_type = response.headers.get("Content-Type")
    #     if content_type and "application/json" in content_type:
    #         js = await response.json()
    #     elif content_type and "text/plain" in content_type:
    #         js = {"message": await response.text()}
    #     else:
    #         raise HTTPException(
    #             status_code=502,
    #             detail=f"The returned data type from the route service is not supported. "
    #             f"Received content type: {content_type}",
    #         )
    #     try:
    #         response.raise_for_status()
    #     except aiohttp.ClientResponseError as e:
    #         detail = js.get("error", {}).get("message", e.message) if "error" in js else js
    #         raise HTTPException(status_code=e.status, detail=detail)
    #     return js
    # {
        #   "candidates": [
        #     {
        #       "output": "Once upon a time, there was a young girl named Lily...",
        #       "safetyRatings": [
        #         {
        #           "category": "HARM_CATEGORY_DEROGATORY",
        #           "probability": "NEGLIGIBLE"
        #         }, ...
        #       ]
        #     {
        #       "output": "Once upon a time, there was a young boy named Billy...",
        #       "safetyRatings": [
        #           ...
        #       ]
        #     }
        #   ]
        # }
    if("text-bison-32k" in path):
        response = _aiohttp_post_bison(base_url, path, payload)
        text ={"candidates": [ 
            {
                "output":response.text
            },
        ]
        }
        return text
    elif("gecko" in path):
        response = _aiohttp_post_gecko(base_url, path, payload)

        return response


async def send_stream_request(
    headers: Dict[str, str], base_url: str, path: str, payload: Dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    """
    Send an HTTP request to a specific URL path with given headers and payload.

    Args:
        headers: The headers to include in the request.
        base_url: The base URL where the request will be sent.
        path: The specific path of the URL to which the request will be sent.
        payload: The payload (or data) to be included in the request.

    Returns:
        The server's response as a JSON object.

    Raises:
        HTTPException if the HTTP request fails.
    """
    async with _aiohttp_post(headers, base_url, path, payload) as response:
        async for line in response.content:
            yield line


def rename_payload_keys(payload: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Rename payload keys based on the specified mapping. If a key is not present in the
    mapping, the key and its value will remain unchanged.

    Args:
        payload: The original dictionary to transform.
        mapping: A dictionary where each key-value pair represents a mapping from the old
            key to the new key.

    Returns:
        A new dictionary containing the transformed keys.

    """
    return {mapping.get(k, k): v for k, v in payload.items()}
