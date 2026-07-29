from utils.s3_downloader import S3ModelDownloader

downloader = S3ModelDownloader()

downloader.download()

print()

print("Artifacts downloaded successfully")