import os
import sys
import time
import zipfile
import tarfile
import shutil
import urllib.request
import urllib.error


DEFAULT_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/dschettler8845/brats-2021-task1"
)
DEFAULT_TARGET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)


def format_bytes(bytes_count):
    if bytes_count >= 1024 ** 3:
        return f"{bytes_count / (1024 ** 3):.2f} GB"
    if bytes_count >= 1024 ** 2:
        return f"{bytes_count / (1024 ** 2):.2f} MB"
    if bytes_count >= 1024:
        return f"{bytes_count / 1024:.2f} KB"
    return f"{bytes_count} bytes"


def download_file(url, target_path, chunk_size=1024 * 1024, max_retries=100):
    """
    Downloads file from URL with resume support, progress logging, and automatic retry.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".part"

    total_size = None
    retries = 0

    while True:
        downloaded = 0
        if os.path.exists(temp_path):
            downloaded = os.path.getsize(temp_path)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"Resuming download from byte {downloaded} ({format_bytes(downloaded)})...", flush=True)

        req = urllib.request.Request(url, headers=headers)

        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range Not Satisfiable (already fully downloaded)
                print("File part already fully downloaded. Renaming to target.")
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_path, target_path)
                return target_path
            print(f"HTTP error {e.code}: {e.reason}. Retrying in 5 seconds...", flush=True)
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Connection error: {e}. Retrying in 5 seconds...", flush=True)
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(5)
            continue

        if total_size is None:
            content_range = resp.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            elif resp.headers.get("Content-Length"):
                total_size = downloaded + int(resp.headers.get("Content-Length"))

        total_str = format_bytes(total_size) if total_size else "Unknown size"
        print(f"Download active: Total size: {total_str}", flush=True)

        start_time = time.time()
        last_log_time = start_time
        bytes_since_last_log = 0

        try:
            with open(temp_path, "ab" if downloaded > 0 else "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        # EOF reached
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    bytes_since_last_log += len(chunk)

                    current_time = time.time()
                    if current_time - last_log_time >= 5.0 or (total_size and downloaded == total_size):
                        speed = bytes_since_last_log / max(current_time - last_log_time, 1e-5)
                        speed_str = f"{format_bytes(int(speed))}/s"
                        if total_size and total_size > 0:
                            pct = (downloaded / total_size) * 100.0
                            remaining_bytes = max(total_size - downloaded, 0)
                            eta_seconds = int(remaining_bytes / max(speed, 1))
                            eta_min = eta_seconds // 60
                            eta_sec = eta_seconds % 60
                            print(
                                f"Progress: {format_bytes(downloaded)} / {total_str} "
                                f"({pct:.1f}%) | Speed: {speed_str} | ETA: {eta_min}m {eta_sec}s",
                                flush=True
                            )
                        else:
                            print(
                                f"Downloaded: {format_bytes(downloaded)} | Speed: {speed_str}",
                                flush=True
                            )
                        last_log_time = current_time
                        bytes_since_last_log = 0

            # If we reached EOF and total size is matched (or not provided)
            if total_size is None or downloaded >= total_size:
                break
        except Exception as e:
            print(f"\nDownload interrupted ({e}). Resuming in 3 seconds...", flush=True)
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(3)
            continue

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"Download complete: {target_path} ({format_bytes(os.path.getsize(target_path))})", flush=True)
    return target_path


def extract_archive(archive_path, extract_to):
    """
    Extracts zip or tar archives, including handling nested archives if present.
    """
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_to}...")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            total_files = len(z.infolist())
            print(f"Archive contains {total_files} items. Unpacking...")
            z.extractall(extract_to)
        print("Zip extraction completed.")
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as t:
            print("Extracting tar archive...")
            t.extractall(extract_to)
        print("Tar extraction completed.")
    else:
        print(f"Warning: {archive_path} is neither a standard zip nor tar file.")
        return

    # Check if there is an inner tar file (often Kaggle datasets contain BraTS2021_Training_Data.tar)
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith(".tar") or f.endswith(".tar.gz"):
                inner_tar = os.path.join(root, f)
                print(f"Found nested tar archive: {inner_tar}. Extracting...")
                try:
                    with tarfile.open(inner_tar, "r:*") as t:
                        t.extractall(root)
                    print(f"Extracted {inner_tar}. Removing inner archive...")
                    os.remove(inner_tar)
                except Exception as ex:
                    print(f"Error extracting inner tar {inner_tar}: {ex}")


def verify_dataset(data_dir):
    """
    Verifies patient folders and modal files.
    """
    print("\nVerifying dataset integrity...")
    patient_dirs = []
    for root, dirs, files in os.walk(data_dir):
        # Check if directory contains flair, t1, t1ce, t2, and seg
        has_nii = any(f.endswith(".nii.gz") for f in files)
        if has_nii:
            modalities = ["_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz", "_flair.nii.gz", "_seg.nii.gz"]
            found = all(any(f.endswith(m) for f in files) for m in modalities)
            if found:
                patient_dirs.append(root)

    print(f"Found {len(patient_dirs)} complete patient directories.")
    if len(patient_dirs) > 0:
        sample = patient_dirs[0]
        print(f"Sample patient: {os.path.basename(sample)} at {sample}")
    return patient_dirs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare BraTS 2021 dataset.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Download URL")
    parser.add_argument("--target_dir", type=str, default=DEFAULT_TARGET_DIR, help="Target extraction directory")
    parser.add_argument("--keep_archive", action="store_true", help="Keep the downloaded zip archive")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing dataset directory")

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.target_dir)
        return

    archive_path = os.path.join(args.target_dir, "BraTS2021_download.zip")
    print(f"Target archive: {archive_path}")
    print(f"Extraction directory: {args.target_dir}")

    download_file(args.url, archive_path)
    extract_archive(archive_path, args.target_dir)

    if not args.keep_archive and os.path.exists(archive_path):
        print(f"Removing downloaded archive {archive_path} to free space...")
        os.remove(archive_path)

    verify_dataset(args.target_dir)
    print("\nDataset preparation finished successfully.")


if __name__ == "__main__":
    main()
