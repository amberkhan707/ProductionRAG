import pytest
from fastapi.testclient import TestClient
from app import app # Import your FastAPI app
from unittest.mock import AsyncMock, patch

# 1. Client Setup
client = TestClient(app)

# 2. Basic Health Check Test
def test_health_check():
    """
    Ye check karega ki server start ho raha hai ya nahi.
    GET / request bhejega aur response 200 OK expect karega.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "pipeline": "Agentic RAG"}

# 3. Chat Endpoint Test (With Mocking)
@pytest.mark.asyncio
async def test_chat_endpoint():
    """
    Ye /chat endpoint test karega bina real LLM/DB ko call kiye.
    Hum 'graph.ainvoke' ko MOCK kar denge.
    """
    
    # Mock Response jo hum chahte hain ki Graph return kare
    mock_response = {
        "generation": "This is a mock answer from the test."
    }

    # 'app.graph.ainvoke' ko patch karein
    # Note: 'app.graph' wahan point karna chahiye jahan graph object app.py mein hai
    with patch("app.graph.ainvoke", new_callable=AsyncMock) as mock_graph:
        mock_graph.return_value = mock_response

        # Request Payload
        payload = {"question": "Test question"}

        # API Call
        response = client.post("/chat", json=payload)

        # Assertions (Checking results)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a mock answer from the test."
        
        # Verify ki mock function call hua tha
        mock_graph.assert_called_once()