import os
import sys
import time

def main():
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("Error: No Synapse auth token provided.")
        sys.exit(1)

    import synapseclient
    target_dir = os.path.abspath("dataset")
    cache_dir = os.path.abspath(".synapseCache")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    syn = synapseclient.Synapse(cache_root_dir=cache_dir)
    print("Logging into Synapse...")
    syn.login(authToken=token, silent=True)
    print("Logged in successfully.")

    entity_id = "syn51514132"
    print(f"Starting download for {entity_id} (ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip)...")
    print(f"Target directory: {target_dir}")

    start_time = time.time()
    try:
        # syn.get with downloadLocation downloads directly to target_dir
        downloaded_entity = syn.get(
            entity_id,
            downloadLocation=target_dir,
            ifcollision="overwrite.local"
        )
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed/60:.2f} minutes!")
        print(f"Downloaded file: {downloaded_entity.path}")
        
        # Check file size
        if os.path.exists(downloaded_entity.path):
            size_gb = os.path.getsize(downloaded_entity.path) / (1024**3)
            print(f"Archive file size: {size_gb:.2f} GB")

    except Exception as e:
        print(f"Error during download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
