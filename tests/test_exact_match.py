"""Unit tests for exact match functionality of SimilarityCache."""

import time
import pytest
from simcache.cache import SimilarityCache


class TestExactMatch:
    """Test cases for exact key matching in the cache."""

    @pytest.mark.asyncio
    async def test_set_and_get_basic(self):
        """Test basic set and get operations."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_and_get_multiple_keys(self):
        """Test setting and getting multiple different keys."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_update_existing_key(self):
        """Test updating an existing key with a new value."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        await cache.set("key1", "updated_value")
        assert cache.get("key1") == "updated_value"

    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        cache = SimilarityCache(threshold=0.8)
        
        result = cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_key(self):
        """Test deleting a key from the cache."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        deleted = cache.delete("key1")
        assert deleted is True
        assert cache.get("key1") is None

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        cache = SimilarityCache(threshold=0.8)
        
        deleted = cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing all entries from the cache."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        assert len(cache) == 3
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cache_size(self):
        """Test cache size tracking."""
        cache = SimilarityCache(threshold=0.8)
        
        assert len(cache) == 0
        
        await cache.set("key1", "value1")
        assert len(cache) == 1
        
        await cache.set("key2", "value2")
        assert len(cache) == 2
        
        cache.delete("key1")
        assert len(cache) == 1

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = SimilarityCache(threshold=0.8, ttl=0.1)  # 100ms TTL
        
        await cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should return None after expiration
        result = cache.get("key1")
        assert result is None
        assert len(cache) == 0  # Expired entry should be removed

    @pytest.mark.asyncio
    async def test_per_item_ttl(self):
        """Test per-item TTL override."""
        cache = SimilarityCache(threshold=0.8, ttl=3600)  # Long default TTL
        
        # Set with short per-item TTL
        await cache.set("key1", "value1", ttl=0.1)
        assert cache.get("key1") == "value1"
        
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_no_ttl_expiration(self):
        """Test that entries without TTL don't expire."""
        cache = SimilarityCache(threshold=0.8, ttl=None)
        
        await cache.set("key1", "value1")
        
        time.sleep(0.1)
        
        # Should still be available
        assert cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when maxsize is reached."""
        cache = SimilarityCache(threshold=0.8, maxsize=2)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Adding key3 should evict key2 (least recently used)
        await cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"  # Should be there

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self):
        """Test that LRU eviction respects access order."""
        cache = SimilarityCache(threshold=0.8, maxsize=3)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1, then key2
        cache.get("key1")
        cache.get("key2")
        
        # Adding key4 should evict key3 (least recently used)
        await cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") is None
        assert cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_update_moves_to_front(self):
        """Test that updating a key moves it to the front (most recently used)."""
        cache = SimilarityCache(threshold=0.8, maxsize=2)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Update key1, which should make it most recently used
        await cache.set("key1", "updated_value1")
        
        # Adding key3 should evict key2, not key1
        await cache.set("key3", "value3")
        
        assert cache.get("key1") == "updated_value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_get_moves_to_front(self):
        """Test that getting a key moves it to the front (most recently used)."""
        cache = SimilarityCache(threshold=0.8, maxsize=2)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Get key1, which should make it most recently used
        cache.get("key1")
        
        # Adding key3 should evict key2, not key1
        await cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_unlimited_size(self):
        """Test that cache works with unlimited size."""
        cache = SimilarityCache(threshold=0.8, maxsize=None)
        
        # Add many keys
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")
        
        assert len(cache) == 100
        
        # All should be retrievable
        for i in range(100):
            assert cache.get(f"key{i}") == f"value{i}"

    @pytest.mark.asyncio
    async def test_different_value_types(self):
        """Test storing different types of values."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("str_key", "string_value")
        await cache.set("int_key", 42)
        await cache.set("list_key", [1, 2, 3])
        await cache.set("dict_key", {"a": 1, "b": 2})
        await cache.set("none_key", None)
        
        assert cache.get("str_key") == "string_value"
        assert cache.get("int_key") == 42
        assert cache.get("list_key") == [1, 2, 3]
        assert cache.get("dict_key") == {"a": 1, "b": 2}
        assert cache.get("none_key") is None

    @pytest.mark.asyncio
    async def test_empty_string_key(self):
        """Test using empty string as key."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"

    @pytest.mark.asyncio
    async def test_none_value(self):
        """Test storing None as a value."""
        cache = SimilarityCache(threshold=0.8)
        
        await cache.set("key1", None)
        result = cache.get("key1")
        assert result is None
