import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import pandas as pd
from config.settings import get_settings
from timescale_vector import client
from ollama import embeddings

class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with the correct database settings and Timescale Vector client."""
        self.settings = get_settings()

        self.embedding_model = self.settings.ollama.embedding_model

        # Connection string for Timescale Vector client
        self.vec_client = client.Sync(
            service_url = self.settings.database.service_url,
            num_dimensions=self.settings.vector_store.embedding_dimensions,
            table_name=self.settings.vector_store.table_name,
            time_partition_interval=timedelta(days=1),
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.

        Raises:
            RuntimeError: If embedding generation fails.
        """
        text = text.replace("\n", " ")
        start_time = time.time()

        try:
            # Call Ollama's embeddings API
            response = embeddings(
                model=self.embedding_model,
                prompt=text,
            )
            embedding = response["embedding"]
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            raise RuntimeError("Embedding generation failed") from e

        elapsed_time = time.time() - start_time
        print(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """
        Create the necessary tables in the database.

        Args:
            embedding_dimensions: The dimensionality of the embeddings.
        """
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search."""
        try:
            self.vec_client.create_embedding_index(client.DiskAnnIndex())
        except Exception:
            print("Index already exists, skipping creation.")

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database."""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
            Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.settings.vector_store.table_name}"
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        # Fix: Use **search_args to pass keyword arguments
        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)]
            , axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )