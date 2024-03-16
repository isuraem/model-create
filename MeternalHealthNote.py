from pydantic import BaseModel

class MeternalHealthNote(BaseModel):
    Age: int
    SystolicBP: int
    DiastolicBP: int
    BS: float
    BodyTemp: int
    HeartRate: int