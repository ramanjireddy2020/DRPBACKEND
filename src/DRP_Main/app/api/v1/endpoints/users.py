from fastapi import APIRouter, HTTPException
from DRP_Main.app.schemas.user import UserCreate, UserRead, UserUpdate
from DRP_Main.app.services.user_service import UserService

router = APIRouter()
user_service = UserService()

@router.post("/", response_model=UserRead)
async def create_user(user: UserCreate):
    return await user_service.create_user(user)

@router.get("/{user_id}", response_model=UserRead)
async def read_user(user_id: int):
    user = await user_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=UserRead)
async def update_user(user_id: int, user: UserUpdate):
    updated_user = await user_service.update_user(user_id, user)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user

@router.delete("/{user_id}", response_model=dict)
async def delete_user(user_id: int):
    success = await user_service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"detail": "User deleted successfully"}