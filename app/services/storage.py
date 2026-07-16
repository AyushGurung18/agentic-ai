import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import uuid

def get_r2_client():
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint = os.getenv("R2_JURISDICTION_SPECIFIC_ENDPOINT")
    
    if not all([access_key, secret_key, endpoint]):
        raise RuntimeError("Missing R2 credentials or endpoint in environment")
    
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
    )
    return client

def upload_file_stream_to_r2(file_obj, filename: str, content_type: str = "application/pdf") -> str:
    """Uploads a file-like object to R2 and returns the public URL."""
    client = get_r2_client()
    bucket_name = os.getenv("R2_BUCKET_NAME", "user-documents-prod")
    
    # Generate unique object key
    unique_id = str(uuid.uuid4())
    object_key = f"uploads/{unique_id}_{filename}"
    
    try:
        # Seek to start just in case
        file_obj.seek(0)
        client.upload_fileobj(
            file_obj,
            bucket_name,
            object_key,
            ExtraArgs={'ContentType': content_type}
        )
    except (BotoCoreError, ClientError) as e:
        print(f"Failed to upload to R2: {e}")
        # In a production app, we might want to log this or handle it,
        # but we don't necessarily want to fail the whole process if upload fails,
        # or maybe we do depending on the requirement.
        raise RuntimeError(f"Failed to upload file to R2: {str(e)}")

    endpoint = os.getenv("R2_JURISDICTION_SPECIFIC_ENDPOINT")
    base = endpoint.replace("https://", "")
    public_url = f"https://{base}/{bucket_name}/{object_key}"

    return public_url


def delete_file_from_r2(r2_url: str) -> None:
    """Best-effort cleanup of an R2 object given its public URL.

    Used when a later step in the upload pipeline fails after the object was
    already durably stored, so we don't leave it orphaned with no DB row.
    """
    bucket_name = os.getenv("R2_BUCKET_NAME", "user-documents-prod")
    object_key = r2_url.split(f"/{bucket_name}/", 1)[-1]
    try:
        client = get_r2_client()
        client.delete_object(Bucket=bucket_name, Key=object_key)
    except (BotoCoreError, ClientError) as e:
        print(f"Failed to clean up orphaned R2 object {object_key}: {e}")
