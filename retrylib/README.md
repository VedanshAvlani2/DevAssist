# retrylib

A small Python utility library providing retry decorators, a simple in-memory cache with TTL support, and common input validators.

## Installation

```bash
pip install -e .
```

## Modules

### `retry`

```python
from retrylib import retry, retry_with_backoff

@retry(max_attempts=3)
def flaky_api_call():
    ...

result = retry_with_backoff(flaky_api_call, max_attempts=5, base_delay=0.5)
```

### `cache`

```python
from retrylib import SimpleCache

cache = SimpleCache(ttl=300)
cache.set("key", "value")
value = cache.get("key")
```

### `validators`

```python
from retrylib import validate_email, validate_phone, validate_url

validate_email("user@example.com")  # True
validate_phone("555-867-5309")      # True
validate_url("https://example.com") # True
```

## Development

This library is intentionally kept small for use as a test target in automated eval harnesses.
