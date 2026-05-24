import os
import yaml
import pandas as pd
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from huggingface_hub import HfApi, hf_hub_download

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

def summarize_hf_yaml_schema(repo_id: str, filename: str) -> pd.DataFrame:
    """
    Downloads a BIRD-formatted schema YAML from Hugging Face, flattens it 
    into a pandas DataFrame, and prints schema summary metrics.
    
    Parameters:
        repo_id (str): The Hugging Face repository ID (e.g., "domyn/FINCH").
        filename (str): The path to the YAML file in the repo.
        
    Returns:
        pd.DataFrame: The flattened underlying schema DataFrame.
    """
    # 1. Download the specific schema file from the repository
    print(f"Downloading '{filename}' from repo '{repo_id}'...")
    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        repo_type="dataset"
    )

    # 2. Parse the YAML file
    print("Parsing YAML data...")
    with open(local_path, 'r', encoding='utf-8') as f:
        schema_data = yaml.safe_load(f)

    # 3. Flatten the nested structure into row dicts
    rows = []
    for db_entry in schema_data:
        db_name = db_entry.get("database")
        
        for key, tables in db_entry.items():
            if key == "database" or not isinstance(tables, dict):
                continue
                
            for table_name, table_body in tables.items():
                # Handle empty tables gracefully
                if not table_body or "columns_info" not in table_body:
                    rows.append({
                        "database": db_name,
                        "table": table_name,
                        "column_name": None
                    })
                    continue
                
                # Extract column fields
                for col in table_body["columns_info"]:
                    rows.append({
                        "database": db_name,
                        "table": table_name,
                        "column_name": col.get("column_name")
                    })

    # 4. Construct DataFrame
    df_schema = pd.DataFrame(rows)

    # 5. Compute Metrics
    num_databases = df_schema["database"].nunique()
    num_tables = df_schema[["database", "table"]].drop_duplicates().shape[0]
    
    # Filter out empty tables so they report a column count of 0 instead of 1
    cols_per_table = (
        df_schema[df_schema["column_name"].notna()]
        .groupby(["database", "table"])["column_name"]
        .count()
        .reset_index(name="column_count")
    )

    # 6. Display Summary Printout
    print("\n" + "="*50)
    print("                SCHEMA SUMMARY")
    print("="*50)
    print(f"1. Total Unique Databases: {num_databases}")
    print(f"2. Total Unique Tables:    {num_tables}")
    print("="*50)
    print("\n3. Number of Columns per Table (Sample preview):")
    print(cols_per_table.head(15).to_string(index=False))
    print("="*50)

    return df_schema


if __name__ == "__main__":
    token = UserSecretsClient().get_secret('HF_TOKEN')
    login(token=token.strip())
    total_remote_files = count_remote_sqlite_files()
    print(f"Total SQLite files found in remote repo: {total_remote_files}")
    df = summarize_hf_yaml_schema(
        repo_id="domyn/FINCH", 
        filename="schemas/database_schemas.yaml"
    )