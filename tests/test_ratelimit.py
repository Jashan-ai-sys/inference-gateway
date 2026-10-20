import pytest

from gateway.ratelimit import HostedLimiter, TokenBucket


def test_bucket_starts_full_and_refills_continuously():
    bucket = TokenBucket(per_minute=60)
    assert bucket.available(0.0) == 60
    bucket.take(60, 0.0)
    assert bucket.available(0.0) == 0
    assert bucket.available(10.0) == pytest.approx(10)


def test_bucket_is_capped_at_one_minute():
    bucket = TokenBucket(per_minute=60)
    bucket.take(1, 0.0)
    assert bucket.available(1000.0) == 60


def test_limiter_checks_requests_and_tokens():
    limiter = HostedLimiter(rpm=2, tpm=1000)
    assert limiter.has_room(600, 0.0)
    limiter.take(600, 0.0)
    assert not limiter.has_room(600, 0.0)  # token budget
    assert limiter.has_room(300, 0.0)
    limiter.take(300, 0.0)
    assert not limiter.has_room(10, 0.0)  # request budget (2 per minute)


def test_limiter_without_limits_always_has_room():
    assert HostedLimiter(rpm=None, tpm=None).has_room(10**9, 0.0)
