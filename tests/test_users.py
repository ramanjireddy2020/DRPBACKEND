# import pytest
# from fastapi.testclient import TestClient
# from app.main import app

# client = TestClient(app)

# def test_create_user():
#     response = client.post("/api/v1/users/", json={"username": "testuser", "email": "test@example.com", "password": "testpassword"})
#     assert response.status_code == 201
#     assert response.json()["username"] == "testuser"

# def test_read_user():
#     response = client.get("/api/v1/users/testuser")
#     assert response.status_code == 200
#     assert response.json()["username"] == "testuser"

# def test_update_user():
#     response = client.put("/api/v1/users/testuser", json={"email": "newemail@example.com"})
#     assert response.status_code == 200
#     assert response.json()["email"] == "newemail@example.com"

# def test_delete_user():
#     response = client.delete("/api/v1/users/testuser")
#     assert response.status_code == 204

# def test_user_not_found():
#     response = client.get("/api/v1/users/nonexistentuser")
#     assert response.status_code == 404