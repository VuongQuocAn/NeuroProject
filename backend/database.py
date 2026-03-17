import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Lấy URL kết nối từ biến môi trường trong docker-compose
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password123@db:5432/neuro_db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency để lấy DB session cho các API sau này
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()