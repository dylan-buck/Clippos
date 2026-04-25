from typing import Literal

from pydantic import BaseModel, ConfigDict


AspectRatio = Literal["9:16", "1:1", "16:9"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
