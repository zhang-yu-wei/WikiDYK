from pathlib import Path
from huggingface_hub import HfApi, Repository

api = HfApi()
slug = "YWZBrandon/wikidyk"

for model_path in Path("all_models").iterdir():
    if not model_path.is_dir(): continue
    repo_id = f"YWZBrandon/{model_path.name}"
    # 1. Create or ensure remote repo exists
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)      
    # 2. Upload folder directly (skips local git clone)
    api.upload_folder(repo_id=repo_id, folder_path=str(model_path), repo_type="model")
    # 3. Add to Collection
    api.add_collection_item(collection_slug=slug, item_id=repo_id, item_type="model")
    print(f"Pushed & added {repo_id}")
