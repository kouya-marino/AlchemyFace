import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer, String, Float
from sqlalchemy.dialects.postgresql import TIMESTAMP as timestamp
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class FaceData(Base):
    __tablename__ = "faceinfo"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String(100), nullable=False)
    group_id = Column(Integer, nullable=False)
    feature_vector = Column(Vector(128), nullable=False)

class EventData(Base):
    __tablename__ = "eventlog"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    detect_time = Column(timestamp, nullable=False)
    face_id = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)

if __name__ == "__main__":
    engine = create_engine(os.environ.get("DATABASE_URL", "postgresql://localhost:5432/faces"))
    Base.metadata.create_all(bind=engine)