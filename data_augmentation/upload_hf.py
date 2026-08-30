# upload_to_hf.py
from pathlib import Path
import os
from huggingface_hub import HfApi
import dotenv

dotenv.load_dotenv()
REPO_ID = "UlyssesLynne/urban_planning_llm"
FOLDER = Path(__file__).resolve().parent / "../instruction_dataset_final_3"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN to a Hugging Face write token before running.")

    if not FOLDER.exists():
        raise SystemExit(f"Missing folder: {FOLDER}")

    api = HfApi(token=token)
    print(f"Uploading JSON files from {FOLDER} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="",  # place at dataset root
        allow_patterns=["*.json"],
        commit_message="Add final instruction JSON files",
    )
    print("Done.")


if __name__ == "__main__":
    main()
