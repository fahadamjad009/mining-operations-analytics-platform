from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

EquipmentType = Literal["haul_truck", "excavator", "drill"]
EventType = Literal["running", "idle", "down"]
Shift = Literal["day", "night"]

class TelemetryRow(BaseModel):
    ts: datetime
    site: str
    equipment_id: str
    equipment_type: EquipmentType
    shift: Shift
    engine_temp_c: float = Field(ge=-20, le=180)
    vibration_mm_s: float = Field(ge=0, le=200)
    fuel_rate_lph: float = Field(ge=0, le=1000)
    speed_kmh: float = Field(ge=0, le=120)
    event: EventType

class MaintenanceRow(BaseModel):
    ts: datetime
    site: str
    equipment_id: str
    work_order_id: str
    fault_code: str
    severity: Literal["low", "medium", "high"]
    downtime_minutes: int = Field(ge=0, le=1440)
