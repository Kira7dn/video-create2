import os
import uuid
import json
import logging
from filelock import FileLock
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from app.presentation.api.v1.schemas.video import JobQueuedResponse
from app.core.config import settings
from app.application.use_cases.video_create import CreateVideoUseCase
from app.presentation.api.v1.dependencies.video import get_create_video_use_case
from utils.resource_manager import managed_temp_directory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos")

# Job store paths
JOB_STORE_PATH = os.path.join("data", "job_store.json")
JOB_STORE_LOCK_PATH = os.path.join("data", "job_store.json.lock")


def load_job_store():
    if not os.path.exists(JOB_STORE_PATH):
        return {}
    with FileLock(JOB_STORE_LOCK_PATH, timeout=5):
        try:
            with open(JOB_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def save_job_store(job_store):
    os.makedirs(os.path.dirname(JOB_STORE_PATH), exist_ok=True)
    with FileLock(JOB_STORE_LOCK_PATH, timeout=5):
        with open(JOB_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(job_store, f)


@router.post("", response_model=JobQueuedResponse)
async def create_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_case: CreateVideoUseCase = Depends(get_create_video_use_case),
):
    """Queue a job to create a video from uploaded JSON configuration."""
    job_id = str(uuid.uuid4())

    # Initialize job status
    job_store = load_job_store()
    job_store[job_id] = {"status": "pending", "result": None, "error": None}
    save_job_store(job_store)

    # Read once to avoid re-reading UploadFile in background task
    content = await file.read()
    filename = file.filename or ""

    async def process_job(file_bytes: bytes, filename: str, job_id: str):
        try:
            # Validate filename and extension
            if not filename:
                raise ValueError("No filename provided")
            allowed = settings.allowed_extensions
            if not any(filename.endswith(ext) for ext in allowed):
                raise ValueError(f"Invalid file format. Allowed: {', '.join(allowed)}")

            # Read and size-check
            if len(file_bytes) > settings.max_file_size:
                raise ValueError(
                    f"File too large. Max size: {settings.max_file_size} bytes"
                )

            # Parse JSON
            json_data = json.loads(file_bytes.decode("utf-8"))
            if not isinstance(json_data, dict) or "segments" not in json_data:
                raise ValueError("Invalid JSON format: 'segments' key is required")

            # Run use case within a managed temp directory specific to this job
            async with managed_temp_directory(
                prefix=settings.temp_batch_dir + "_"
            ) as temp_dir:
                # Re-compose use case with job-scoped temp_dir so infra writes there
                uc_scoped = get_create_video_use_case(temp_dir=temp_dir)
                result = await uc_scoped.execute(json_data)

            job_store = load_job_store()
            job_store[job_id]["status"] = "done"
            job_store[job_id]["result"] = result.get("s3_url")
            save_job_store(job_store)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            job_store = load_job_store()
            job_store[job_id]["status"] = "failed"
            job_store[job_id]["error"] = str(e)
            save_job_store(job_store)

    background_tasks.add_task(process_job, content, filename, job_id)
    return JobQueuedResponse(job_id=job_id)


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job_store = load_job_store()
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "Job not found"})
    return job
