# import pytest
# from app.models.item import Item
# from app.schemas.item import ItemCreate, ItemUpdate

# @pytest.fixture
# def item_data():
#     return {
#         "name": "Test Item",
#         "description": "This is a test item.",
#         "price": 10.99,
#         "quantity": 5
#     }

# def test_create_item(item_data):
#     item_create = ItemCreate(**item_data)
#     item = Item(**item_create.dict())
#     assert item.name == item_data["name"]
#     assert item.description == item_data["description"]
#     assert item.price == item_data["price"]
#     assert item.quantity == item_data["quantity"]

# def test_update_item(item_data):
#     item_create = ItemCreate(**item_data)
#     item = Item(**item_create.dict())
    
#     update_data = {"name": "Updated Item", "price": 12.99}
#     item_update = ItemUpdate(**update_data)
    
#     for key, value in item_update.dict(exclude_unset=True).items():
#         setattr(item, key, value)
    
#     assert item.name == update_data["name"]
#     assert item.price == update_data["price"]
#     assert item.description == item_data["description"]  # Should remain unchanged
#     assert item.quantity == item_data["quantity"]  # Should remain unchanged