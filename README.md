# SimCache

A powerful in-memory similarity cache with LRU eviction and TTL expiration, designed for applications that need to cache and retrieve values based on semantic similarity of keys.

## Features

- **Semantic Similarity Search**: Uses sentence transformers to find similar keys even when exact matches don't exist
- **LRU Eviction**: Automatically evicts least recently used items when the cache reaches its maximum size
- **TTL Expiration**: Supports time-to-live expiration for cache entries (global or per-item)
- **Async Operations**: Non-blocking embedding generation for better performance
- **Exact Match Priority**: Exact key matches are always preferred over similarity matches

## Installation

```bash
pip install similaritycache
```

Or using `uv`:

```bash
uv add similaritycache
```

## Requirements

- Python >= 3.13
- numpy >= 2.3.5
- sentence-transformers >= 5.2.0

## Quick Start

```python
import asyncio
from simcache import SimilarityCache

async def main():
    # Create a cache with similarity threshold of 0.8
    cache = SimilarityCache(threshold=0.8)
    
    # Store a value
    await cache.set("What is machine learning?", "ML is a subset of AI")
    
    # Retrieve by exact match
    result = cache.get("What is machine learning?")
    print(result)  # "ML is a subset of AI"
    
    # Retrieve by similar query (if similarity is high enough)
    result = cache.get("What is ML?")
    print(result)  # "ML is a subset of AI" (if similarity >= threshold)

asyncio.run(main())
```

## Configuration Options

### `SimilarityCache` Constructor Parameters

#### `threshold` (required)
- **Type**: `float`
- **Description**: Similarity threshold for matching keys. The cache will return the best matching entry if its similarity score meets or exceeds this threshold.
- **Range**: Typically between 0.0 and 1.0 (cosine similarity)
- **Example**: `threshold=0.8` means only matches with 80%+ similarity will be returned

#### `model_name` (optional)
- **Type**: `str`
- **Default**: `"all-MiniLM-L6-v2"`
- **Description**: The sentence transformer model to use for generating embeddings. You can use any model compatible with `sentence-transformers`.
- **Examples**:
  - `"all-MiniLM-L6-v2"` - Fast, lightweight (default)
  - `"all-mpnet-base-v2"` - Better quality, slower
  - `"paraphrase-multilingual-MiniLM-L12-v2"` - Multilingual support

#### `maxsize` (optional)
- **Type**: `int | None`
- **Default**: `None` (unlimited)
- **Description**: Maximum number of items the cache can hold. When the limit is reached, the least recently used (LRU) item is automatically evicted.
- **Example**: `maxsize=100` limits the cache to 100 items

#### `ttl` (optional)
- **Type**: `float | None`
- **Default**: `None` (no expiration)
- **Description**: Default time-to-live in seconds for cache entries. Entries expire after this duration and are automatically removed.
- **Example**: `ttl=3600` sets entries to expire after 1 hour

## Usage Examples

### Basic Usage

```python
import asyncio
from simcache import SimilarityCache

async def basic_example():
    cache = SimilarityCache(threshold=0.7)
    
    # Store values
    await cache.set("question1", "answer1")
    await cache.set("question2", "answer2")
    
    # Retrieve values
    print(cache.get("question1"))  # "answer1"
    print(cache.get("question2"))  # "answer2"
    print(cache.get("nonexistent"))  # None

asyncio.run(basic_example())
```

### Similarity Search

```python
import asyncio
from simcache import SimilarityCache

async def similarity_example():
    cache = SimilarityCache(threshold=0.7)
    
    # Store a question-answer pair
    await cache.set("What is machine learning?", "ML is a subset of AI")
    
    # Wait a moment for embedding to be generated
    await asyncio.sleep(0.5)
    
    # Similar queries can retrieve the same answer
    result1 = cache.get("What is ML?")
    result2 = cache.get("Explain machine learning")
    
    # Both should return "ML is a subset of AI" if similarity is high enough
    print(result1, result2)

asyncio.run(similarity_example())
```

### LRU Eviction

```python
import asyncio
from simcache import SimilarityCache

async def lru_example():
    # Create cache with max size of 2
    cache = SimilarityCache(threshold=0.8, maxsize=2)
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Access key1 to make it more recently used
    cache.get("key1")
    
    # Adding key3 will evict key2 (least recently used)
    await cache.set("key3", "value3")
    
    print(cache.get("key1"))  # "value1" - still there
    print(cache.get("key2"))  # None - evicted
    print(cache.get("key3"))  # "value3" - new entry

asyncio.run(lru_example())
```

### TTL Expiration

```python
import asyncio
import time
from simcache import SimilarityCache

async def ttl_example():
    # Create cache with 1 second TTL
    cache = SimilarityCache(threshold=0.8, ttl=1.0)
    
    await cache.set("key1", "value1")
    print(cache.get("key1"))  # "value1"
    
    # Wait for expiration
    time.sleep(1.5)
    
    print(cache.get("key1"))  # None - expired and removed

asyncio.run(ttl_example())
```

### Per-Item TTL

```python
import asyncio
import time
from simcache import SimilarityCache

async def per_item_ttl_example():
    # Cache with long default TTL
    cache = SimilarityCache(threshold=0.8, ttl=3600)
    
    # Set item with short TTL
    await cache.set("key1", "value1", ttl=0.5)
    await cache.set("key2", "value2")  # Uses default TTL
    
    time.sleep(0.6)
    
    print(cache.get("key1"))  # None - expired
    print(cache.get("key2"))  # "value2" - still valid

asyncio.run(per_item_ttl_example())
```

### Custom Model

```python
import asyncio
from simcache import SimilarityCache

async def custom_model_example():
    # Use a different embedding model
    cache = SimilarityCache(
        threshold=0.8,
        model_name="all-mpnet-base-v2"  # Better quality, slower
    )
    
    await cache.set("What is AI?", "Artificial Intelligence")
    result = cache.get("What is AI?")
    print(result)

asyncio.run(custom_model_example())
```

### Cache Management

```python
import asyncio
from simcache import SimilarityCache

async def cache_management_example():
    cache = SimilarityCache(threshold=0.8)
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Check cache size
    print(len(cache))  # 2
    
    # Delete specific key
    deleted = cache.delete("key1")
    print(deleted)  # True
    print(len(cache))  # 1
    
    # Clear all entries
    cache.clear()
    print(len(cache))  # 0

asyncio.run(cache_management_example())
```

## API Reference

### Methods

#### `get(key: Any) -> Optional[Any]`
Retrieve a value from the cache.

- **Parameters**:
  - `key`: The key to look up (can be any hashable type, but similarity search works best with strings)
- **Returns**: The value if found (exact match or similarity match), `None` otherwise
- **Behavior**:
  1. First checks for exact key match
  2. If no exact match, performs similarity search using embeddings
  3. Returns the best matching entry if similarity meets threshold
  4. Updates LRU order on access

#### `set(key: Any, value: Any, ttl: Optional[float] = None) -> None`
Add or update a value in the cache.

- **Parameters**:
  - `key`: The key to store
  - `value`: The value to store (can be any type)
  - `ttl`: Optional per-item TTL override (uses default if `None`)
- **Returns**: `None`
- **Behavior**:
  - If key exists, updates value and TTL
  - If key doesn't exist, creates new entry
  - Triggers async embedding generation for the key
  - Evicts LRU item if `maxsize` is reached

#### `delete(key: Any) -> bool`
Remove a specific key from the cache.

- **Parameters**:
  - `key`: The key to remove
- **Returns**: `True` if key was found and removed, `False` otherwise

#### `clear() -> None`
Remove all entries from the cache.

- **Returns**: `None`

#### `__len__() -> int`
Return the number of items in the cache.

- **Returns**: Current cache size

## How It Works

1. **Embedding Generation**: When you `set` a key, the cache asynchronously generates an embedding using the specified sentence transformer model.

2. **Exact Match First**: When you `get` a key, the cache first checks for an exact match. This is fast and always preferred.

3. **Similarity Search**: If no exact match is found, the cache:
   - Generates an embedding for the query key
   - Compares it with all cached key embeddings using cosine similarity
   - Returns the best match if similarity meets the threshold

4. **LRU Management**: The cache maintains a doubly-linked list to track access order. Most recently accessed items are moved to the front, and least recently used items are evicted when the cache is full.

5. **TTL Expiration**: Expired entries are automatically removed during `get` and `set` operations.

## Performance Considerations

- **Embedding Generation**: Embeddings are generated asynchronously, so `set` operations return immediately. However, similarity search requires embeddings to be ready.
- **Model Selection**: Smaller models (like `all-MiniLM-L6-v2`) are faster but may have lower quality. Larger models provide better similarity matching but are slower.
- **Cache Size**: With `maxsize=None`, the cache can grow unbounded. Consider setting a reasonable limit for production use.
- **Similarity Search**: Similarity search scans all entries, so performance degrades with cache size. Consider using `maxsize` to limit growth.

## License

This project is open source. See the license file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
