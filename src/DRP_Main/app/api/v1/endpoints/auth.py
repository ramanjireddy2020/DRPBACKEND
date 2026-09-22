from fastapi import APIRouter, Depends
from DRP_Main.app.schemas.user import UserCreate, UserOut
from DRP_Main.app.services.user_service import create_user, get_current_user

router = APIRouter()

@router.post("/login", response_model=UserOut)
async def login(user: UserCreate):
    # Logic for user login
    pass

@router.post("/token")
async def generate_token(user: UserCreate):
    # Logic for token generation
    pass

@router.get("/users/me", response_model=UserOut)
async def read_users_me(current_user: UserOut = Depends(get_current_user)):
    return current_user