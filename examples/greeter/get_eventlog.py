import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from face_recognition.data_capsule import EventData
import json
import datetime as dt

def entry():
    engine = create_engine(os.environ.get("DATABASE_URL", "postgresql://localhost:5432/faces"))
    SessionClass = sessionmaker(engine)
    session = SessionClass()
    event_items = session.query(EventData).all()
    logdata = []
    for event_item in event_items:
        localdict = {"id": event_item.id, "detect_time": dt.datetime.strftime(event_item.detect_time, "%Y/%m/%d %H:%M:%S.%f")[:-3], "face_id": event_item.face_id, "score": event_item.score}
        logdata.append(localdict)
    print(json.dumps(logdata))

if __name__ == "__main__":
    entry()
#    pass