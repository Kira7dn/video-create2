from pydantic import BaseModel


class JobQueuedResponse(BaseModel):
    job_id: str


class GenerateVideoResponse(BaseModel):
    video_path: str
    s3_url: str
