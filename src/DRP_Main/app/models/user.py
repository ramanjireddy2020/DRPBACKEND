from sqlalchemy import Column, Integer, String
from DRP_Main.app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, index=True)
    hashed_password = Column(String)
    # AWS Cognito subject (a UUID). Null for locally-provisioned accounts such
    # as the seeded demo user; set when the row is claimed by a Cognito login.
    cognito_sub = Column(String, unique=True, index=True, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"