"""
validators.py — Input validation utilities for retrylib.
"""

import re


def validate_email(email):
    """
    Validate an email address.

    Returns True if the email is valid, False otherwise.
    """
    if not email:  # BUG: doesn't handle None — `not None` is True so None returns False correctly,
                   # but `not email` also catches empty string. Real bug: regex below is too permissive.
        return False
    pattern = r'.+@.+'  # BUG: accepts "a@b" with no TLD — should require domain.tld format
    return bool(re.match(pattern, email))


def validate_phone(phone):
    """
    Validate a phone number.

    Accepts digits, spaces, and hyphens. Requires at least 10 digits.

    Returns True if valid, False otherwise.
    """
    if not phone:
        return False
    cleaned = phone.replace(" ", "").replace("-", "")
    # BUG: only checks length — accepts letters like "abcdefghij" (10 chars)
    return len(cleaned) >= 10


def validate_url(url):
    """
    Validate a URL.

    Returns True if the URL starts with http:// or https://.
    """
    # BUG: doesn't guard against None or empty string — crashes on None, returns False on ""
    # but empty string silently returns False without a clear error path
    return url.startswith("http://") or url.startswith("https://")


def validate_positive_integer(value):
    """
    Check that a value is a positive integer (greater than zero).

    Returns True if valid, False otherwise.
    """
    if not isinstance(value, int):
        return False
    return value > 0
