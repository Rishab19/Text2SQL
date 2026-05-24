from huggingface_hub import login
from huggingface_hub import HfApi

def count_remote_sqlite_files(repo_id="domyn/FINCH",verbose = False):
    api = HfApi()
    
    # List all files in the dataset repository
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # Common extensions for SQLite databases
    db_extensions = ('.db', '.sqlite', '.sqlite3')
    
    sqlite_files = [f for f in repo_files if f.lower().endswith(db_extensions)]
    
    # Optional: print out the discovered paths
    if verbose:
        for f in sqlite_files:
            print(f"Found remote file: {f}")
        
    return len(sqlite_files)



if __name__ == "__main__":
    token = input("Please input your Hugging Face token to access the repository: ")
    login(token=token.strip())
    total_remote_files = count_remote_sqlite_files()
    print(f"Total SQLite files found in remote repo: {total_remote_files}")