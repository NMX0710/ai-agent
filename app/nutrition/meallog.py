from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"
    unknown = "unknown"


class InputSource(str, Enum):
    text = "text"
    image = "image"
    mixed = "mixed"
    manual = "manual"


class MealLogStatus(str, Enum):
    draft = "draft"
    confirmed = "confirmed"
    synced = "synced"
    sync_failed = "sync_failed"


class SyncStatus(str, Enum):
    pending = "pending"
    synced = "synced"
    failed = "failed"
    skipped = "skipped"


class NutrientValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0)
    unit: str = Field(min_length=1, max_length=20)


class NutritionTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy_kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)

    fiber_g: Optional[float] = Field(default=None, ge=0)
    sugar_g: Optional[float] = Field(default=None, ge=0)
    sodium_mg: Optional[float] = Field(default=None, ge=0)
    cholesterol_mg: Optional[float] = Field(default=None, ge=0)
    water_ml: Optional[float] = Field(default=None, ge=0)

    micronutrients: Dict[str, NutrientValue] = Field(default_factory=dict)


class MealItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    amount: Optional[float] = Field(default=None, gt=0)
    unit: Optional[str] = Field(default=None, min_length=1, max_length=30)
    estimated: bool = True
    brand: Optional[str] = Field(default=None, max_length=200)
    nutrition: Optional[NutritionTotals] = None
    source_ref: Optional[str] = Field(default=None, max_length=500)


class ConfidenceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: float = Field(ge=0, le=1)
    nutrition: Optional[float] = Field(default=None, ge=0, le=1)
    item_recognition: Optional[float] = Field(default=None, ge=0, le=1)


class RawInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Optional[str] = Field(default=None, max_length=100)
    message_id: Optional[str] = Field(default=None, max_length=120)
    text: Optional[str] = None
    image_refs: List[str] = Field(default_factory=list)


class ExternalSyncTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sync_status: SyncStatus = SyncStatus.pending
    external_id: Optional[str] = Field(default=None, max_length=300)
    sync_identifier: Optional[str] = Field(default=None, max_length=300)
    sync_version: Optional[int] = Field(default=None, ge=1)
    last_error: Optional[str] = None
    synced_at: Optional[datetime] = None


class CanonicalMealLog(BaseModel):
    """
    Platform-neutral meal record model.
    Connector-specific payloads (Apple Health, other apps) should be derived from this model.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "meal_log.v1"
    meal_id: UUID
    user_id: str = Field(min_length=1, max_length=200)
    consumed_at: datetime
    timezone: str = Field(min_length=1, max_length=80)
    meal_type: MealType = MealType.unknown
    input_source: InputSource
    status: MealLogStatus = MealLogStatus.draft
    record_version: int = Field(default=1, ge=1)

    nutrition_totals: NutritionTotals
    items: List[MealItem] = Field(default_factory=list)
    confidence: ConfidenceInfo
    raw_inputs: Optional[RawInputs] = None
    external_sync: Dict[str, ExternalSyncTarget] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != "meal_log.v1":
            raise ValueError("schema_version must be 'meal_log.v1'")
        return value

    @field_validator("consumed_at")
    @classmethod
    def _validate_consumed_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("consumed_at must include timezone info")
        return value
