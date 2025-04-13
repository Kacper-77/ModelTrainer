from .database import Base
from sqlalchemy import Integer, String, Column, LargeBinary, Text, JSON, Boolean, ForeignKey


class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True)
    username = Column(String, unique=True)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String)
    phone_number = Column(String)


class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_file = Column(LargeBinary, nullable=False)
    metrics = Column(JSON)
    target_column = Column(String, nullable=True)
    training_data_path = Column(String, nullable=True)
    status = Column(String, default="trained")
    hyperparameters = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
