from typing import Any, Dict
from langchain_voyageai import VoyageAIEmbeddings
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

project_root = Path(__file__).resolve().parents[4]
conf_path = str(Path(project_root) / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


def make_api_call(chunks_df: pd.DataFrame, model_name: str) -> list:
    """
    Compute embeddings for the chunks of text using the Voyage AI API.

    Parameters:
    ----------
    chunks_df: pd.DataFrame
        Dataframe containing chunks of text.
    api_key: str
        API key for the Voyage AI API.

    Returns:
    ----------
    pd.DataFrame
        Dataframe containing the computed embeddings.
    """
    api_key = credentials["voyage_api_key"]
    model = VoyageAIEmbeddings(voyage_api_key=api_key, model=model_name, batch_size=128)
    documents = chunks_df["text"].to_list()

    embeddings = model.embed_documents(documents)

    return pd.DataFrame(embeddings, index=chunks_df.index)


def compute_pca_projection(
    embeddings: pd.DataFrame, pca_config: Dict[str, Any]
) -> pd.DataFrame:

    # Find the number of components to keep the given explained variance ratio
    pca = PCA().fit(embeddings.to_numpy())

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = (
        np.argmax(explained_variance > pca_config["explained_variance_ratio"]) + 1
    )

    pc = PCA(n_components=n_components).fit_transform(embeddings)

    return pd.DataFrame(pc, index=embeddings.index)


def compute_umap_projection(
    pca_projection: pd.DataFrame,
    umap_config: Dict[str, Any] = dict(
        n_neighbors=15, n_components=2, metric="cosine", min_dist=0.1
    ),
) -> pd.DataFrame:
    """
    Compute UMAP projection for the computed embeddings.

    Parameters:
    ----------
    chunks_df: pd.DataFrame
        Dataframe containing the computed embeddings.

    Returns:
    ----------
    pd.DataFrame
        Dataframe containing the UMAP projection.
    """
    from umap import UMAP

    pc = pca_projection.to_numpy()

    umap_projection = UMAP(**umap_config).fit_transform(pc)

    return pd.DataFrame(umap_projection, index=pca_projection.index)
