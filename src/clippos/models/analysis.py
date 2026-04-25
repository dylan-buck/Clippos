from typing import Annotated

from pydantic import Field

from clippos.models.media import ContractModel


class MediaProbe(ContractModel):
    duration_seconds: Annotated[float, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    fps: Annotated[float, Field(gt=0)]
    audio_sample_rate: Annotated[int, Field(gt=0)]
