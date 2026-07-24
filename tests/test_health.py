"""
Health endpoint tests.
"""

from fastapi.testclient import TestClient

from app import app


# ============================================================
# Create test client
# ============================================================

client = TestClient(app)


# ============================================================
# Test health endpoint
# ============================================================


def test_health_endpoint():
    """
    Verify that the health endpoint is available.
    """

    response = client.get("/health")

    assert response.status_code == 200


# ============================================================
# Verify health response
# ============================================================


def test_health_response():
    """
    Verify that the health endpoint returns JSON.
    """

    response = client.get("/health")

    assert response.headers["content-type"].startswith("application/json")

    assert response.json() is not None
