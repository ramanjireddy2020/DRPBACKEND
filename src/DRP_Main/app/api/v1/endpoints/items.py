from fastapi import APIRouter, HTTPException
from DRP_Main.app.schemas.item import ItemCreate, ItemUpdate, Item
from DRP_Main.app.services.item_service import ItemService

router = APIRouter()
item_service = ItemService()

@router.post("/", response_model=Item)
async def create_item(item: ItemCreate):
    return await item_service.create_item(item)

@router.get("/{item_id}", response_model=Item)
async def read_item(item_id: int):
    item = await item_service.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.put("/{item_id}", response_model=Item)
async def update_item(item_id: int, item: ItemUpdate):
    updated_item = await item_service.update_item(item_id, item)
    if not updated_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated_item

@router.delete("/{item_id}", response_model=dict)
async def delete_item(item_id: int):
    success = await item_service.delete_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"detail": "Item deleted successfully"}