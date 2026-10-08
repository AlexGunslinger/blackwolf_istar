from numpy import full
import boto3
import os
from pathlib import Path
from glob import glob
from .s3_bucket_config import ACCESS_KEY, SECRET_KEY, BUCKET_NAME

def s3_download_directory(src, dst, data_type='.e57', sub_folders=False):
    # destination location
    if not os.path.exists(dst):
        os.makedirs(dst)
    # aws resource: s3
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    # blackwolf bucket
    bucket = s3.Bucket(BUCKET_NAME)
    # source directory
    src_dirname = Path(src).stem
    # find resources that match the input path
    for obj in bucket.objects.filter(Prefix=src):
        path = Path(obj.key)
        if path.suffix == data_type:
            if not sub_folders:
                # check if this resource is in a subfolder
                if path.parent.stem != src_dirname:
                    continue
            rel_path = path.relative_to(Path(src))
            dst_dir = os.path.join(dst, rel_path.parent)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            bucket.download_file(obj.key, os.path.join(dst_dir, path.name))

def s3_upload_directory(src, dst):
    # aws resource: s3
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    # 
    for root, _, files in os.walk(src):
        for f in files:
            full_path = os.path.join(root, f).replace('\\', '/')
            key = os.path.join(
                dst,
                Path(src).stem,
                os.path.relpath(full_path, src)
            ).replace('\\', '/')
            print(f'Uploading {key}')
            s3.meta.client.upload_file(
                full_path,
                BUCKET_NAME,
                key
            )
if __name__=="__main__":
    s3_download_directory(
        'Auto Registration Project/JB_GUI_Test', 
        'D:\REGISTRATION DATA\Test_S3\JB_GUI_Test'
    )
    s3_upload_directory(
        'D:\REGISTRATION DATA\Test_S3\JB_GUI_Test',
        'Auto Registration Project/JB_GUI_Test/Test'
    )