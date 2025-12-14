import asyncio
import time
import numpy as np

from typing import Any, Optional
from sentence_transformers import SentenceTransformer
from simcache.node import Node

class SimilarityCache:
    """Configurable in-memory cache with LRU eviction and TTL expiration."""

    def __init__(
        self,
        threshold: float,
        model_name: str = "all-MiniLM-L6-v2",
        maxsize: Optional[int] = None,
        ttl: Optional[float] = None,
    ):
        """
        Initialize the cache.

        Args:
            threshold: Threshold for similarity search
            model_name: Name of the model to use for similarity search (default: "all-MiniLM-L6-v2")
            maxsize: Maximum number of items (None for unlimited)
            ttl: Default time to live in seconds (None for no expiration)
        """
        self.threshold = threshold
        self.model_name = model_name
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: dict[Any, Node] = {}
        self._head: Optional[Node] = None
        self._tail: Optional[Node] = None
        self._size = 0
        self._encoder = None
        self._embedding_tasks: dict[Any, asyncio.Task[None]] = {}

    def _get_encoder(self) -> Any:
        """Lazy load the encoder."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _embed(self, text: str) -> np.ndarray:
        """Embed a text using the encoder."""
        encoder = self._get_encoder()
        embedding = np.array(encoder.encode(text))
        # Normalize for cosine similarity to be based on direction, not magnitude
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _cosine_similarity(self, query_embedding: Optional[np.ndarray], node_embedding: Optional[np.ndarray]) -> float:
        """Calculate cosine similarity between two vectors."""
        if query_embedding is None or node_embedding is None:
            return 0.0
        return float(np.dot(query_embedding, node_embedding))

    def _move_to_front(self, node: Node) -> None:
        """Move a node to the front (head) of the doubly linked list."""
        if node is self._head:
            return

        # Remove node from its current position
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

        # If node was tail, update tail
        if node is self._tail:
            self._tail = node.prev

        # Move node to head
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node

        # If this was the first node, set it as tail too
        if self._tail is None:
            self._tail = node

    def _evict_lru(self) -> None:
        """Remove the least recently used node (tail) from the cache."""
        if self._tail is None:
            return

        node_to_remove = self._tail
        self._tail = node_to_remove.prev

        if self._tail:
            self._tail.next = None
        else:
            # Cache is now empty
            self._head = None

        # Remove from dictionary
        del self._cache[node_to_remove.key]
        self._size -= 1

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on its TTL."""
        if node.expires_at is None:
            return False
        return time.time() > node.expires_at

    def _clean_expired(self) -> None:
        """Remove all expired entries from the cache."""
        current = self._head
        nodes_to_remove = []

        while current:
            if self._is_expired(current):
                nodes_to_remove.append(current)
            current = current.next

        for node in nodes_to_remove:
            self._remove_node(node)

    def _remove_node(self, node: Node) -> None:
        """Remove a specific node from the doubly linked list and cache."""
        # Update prev node
        if node.prev:
            node.prev.next = node.next
        else:
            # Node is head
            self._head = node.next

        # Update next node
        if node.next:
            node.next.prev = node.prev
        else:
            # Node is tail
            self._tail = node.prev

        # Remove from dictionary
        del self._cache[node.key]
        self._size -= 1

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up

        Returns:
            The value if found and not expired, None otherwise
        """

        # Fast path: exact match
        if key in self._cache:
            node = self._cache[key]
            if not self._is_expired(node):
                self._move_to_front(node)
                return node.value
            self._remove_node(node)
            return None

        # Slow path: similarity search
        query_embedding = self._embed(key)
        best_match = None
        best_similarity = 0.0
        for node in self._cache.values():
            if self._is_expired(node):
                continue
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = node
        
        if best_similarity < self.threshold or best_match is None:
            return None

        self._move_to_front(best_match)
        return best_match.value

    async def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update a value in the cache.

        Args:
            key: The key to store
            value: The value to store
            ttl: Optional per-item TTL override (uses default if None)
        """
        # Clean expired items first
        if self.ttl is not None:
            self._clean_expired()

        # Use per-item TTL if provided, otherwise use default
        item_ttl = ttl if ttl is not None else self.ttl
        expires_at = None
        if item_ttl is not None:
            expires_at = time.time() + item_ttl

        # Check if key already exists
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            # Move to front (most recently used)
            self._move_to_front(node)
            # Trigger embedding of the key
            asyncio.create_task(self._generate_embedding(key))
            return

        # Create new node with None embedding (will be set later)
        node = Node(key, value, None, expires_at)

        # Check if we need to evict
        if self.maxsize is not None and self._size >= self.maxsize:
            self._evict_lru()

        # Add to front
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node

        # If this is the first node, set as tail
        if self._tail is None:
            self._tail = node

        # Add to dictionary
        self._cache[key] = node
        self._size += 1

        # Trigger embedding of the key
        asyncio.create_task(self._generate_embedding(key))
    
    async def _generate_embedding(self, key: Any) -> None:
        """Generate the embedding for a key."""
        if key in self._embedding_tasks:
            self._embedding_tasks[key].cancel()
        
        async def _embed_task() -> None:
            """Embed a key."""
            try:
                embedding = self._embed(key)
                self._cache[key].embedding = embedding
            except Exception as e:
                print(f"Error generating embedding for key {key}: {e}")
                raise e
            finally:
                del self._embedding_tasks[key]
        
        self._embedding_tasks[key] = asyncio.create_task(_embed_task())
    
    def delete(self, key: Any) -> bool:
        """
        Remove a specific key from the cache.

        Args:
            key: The key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        node = self._cache.get(key)
        if node is None:
            return False

        self._remove_node(node)
        return True

    def clear(self) -> None:
        """Remove all entries from the cache."""

        current = self._head
        while current:
            next_node = current.next
            self._remove_node(current)
            current = next_node

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return self._size
