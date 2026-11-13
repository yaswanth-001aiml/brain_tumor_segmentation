import os
import sys

def main():
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("Error: No Synapse auth token provided.")
        sys.exit(1)

    import synapseclient
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    print("Successfully logged into Synapse!")
    
    # Check current user
    profile = syn.getUserProfile()
    print(f"Logged in as: {profile.get('userName')} ({profile.get('firstName', '')} {profile.get('lastName', '')})")

    # Check download cart
    print("\nChecking Synapse Download Cart...")
    try:
        from synapseclient.core.download import download_by_file_handle
        cart = syn.get_download_list()
        print(f"Download cart has files available.")
    except Exception as e:
        print(f"Cart query result: {e}")

    # Inspect BraTS 2023 Challenge Project
    print("\nInspecting BraTS 2023 project (syn51156910)...")
    try:
        children = list(syn.getChildren("syn51156910"))
        print(f"Found {len(children)} top-level items in syn51156910:")
        for child in children:
            print(f" - {child['name']} (ID: {child['id']}, Type: {child['type']})")
    except Exception as e:
        print(f"Error querying syn51156910: {e}")

if __name__ == "__main__":
    main()
