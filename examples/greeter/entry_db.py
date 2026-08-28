import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from face_recognition.data_capsule import FaceData
import pandas as pd
import numpy as np

def entry():
    engine = create_engine(os.environ.get("DATABASE_URL", "postgresql://localhost:5432/faces"))
    SessionClass = sessionmaker(engine)
    session = SessionClass()
    df_id = pd.read_csv("data/id2map2.csv")
    for i in range(len(df_id)):
        name = df_id["name"].iloc[i]
        group_name = df_id["type"].iloc[i]
        if group_name == "staff":
            group_id = 1
        else:
            group_id = 2
#        print(f"{i},{name},{group_id}")
        feature_vector = np.load(f"face_data/{i+43}.npy")
        item = FaceData(id=43, name=name, group_id=group_id, feature_vector=feature_vector.reshape(-1))
        with session.no_autoflush:
            session.add(item)
#        session.commit()

if __name__ == "__main__":
    entry()
#    pass