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
    into a pandas DataFrame using strict root-entry separation, and prints metrics.
    """
    # 1. Download file
    print(f"Downloading '{filename}' from repo '{repo_id}'...")
    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        repo_type="dataset"
    )

    # 2. Parse YAML
    print("Parsing YAML data...")
    with open(local_path, 'r', encoding='utf-8') as f:
        schema_data = yaml.safe_load(f)

    # 3. Flawless Flattening Strategy
    rows = []
    
    # Each item in the top-level list represents an isolated database schema context
    for db_entry in schema_data:
        if not isinstance(db_entry, dict):
            continue
            
        # Extract the true database name
        db_name = db_entry.get("database")
        if not db_name:
            continue
            
        # Look at the other keys inside this specific database block
        for key, value in db_entry.items():
            if key == "database" or not isinstance(value, dict):
                continue
                
            # 'value' contains the table names (e.g., customers, gasstations)
            for table_name, table_body in value.items():
                # Account for empty table shells
                if not isinstance(table_body, dict) or "columns_info" not in table_body:
                    rows.append({
                        "database": db_name,
                        "domain_context": key,
                        "table": table_name,
                        "column_name": None
                    })
                    continue
                
                # Extract the column instances
                for col in table_body["columns_info"]:
                    rows.append({
                        "database": db_name,
                        "domain_context": key,
                        "table": table_name,
                        "column_name": col.get("column_name")
                    })

    # 4. Generate DataFrame
    df_schema = pd.DataFrame(rows)

    # 5. Compute the target metrics precisely
    num_databases = df_schema["database"].nunique()
    
    # Grouping by both keeps identical table names in separate databases unique
    unique_tables_df = df_schema[["database", "table"]].drop_duplicates()
    num_tables = unique_tables_df.shape[0]
    
    # Calculate unique domains/specializations found in the data mapping
    num_domains = df_schema["domain_context"].nunique()
    
    # Calculate column counts per table (0 if empty table)
    cols_per_table = (
        df_schema[df_schema["column_name"].notna()]
        .groupby(["database", "table"])["column_name"]
        .count()
        .reset_index(name="column_count")
    )

    # 6. Beautiful Scannable Console Output
    print("\n" + "="*50)
    print("                FINCH SCHEMA METRICS")
    print("="*50)
    print(f"📊 TOTAL DATABASES:       {num_databases}")
    print(f"📅 FINANCIAL DOMAINS:     {num_domains} specialized areas")
    print(f"🏢 TABLES:                {num_tables} across all databases")
    print("="*50)
    print("\n🔢 COLUMNS PER TABLE (Preview of first 10 rows):")
    print("-" * 50)
    print(cols_per_table.head(10).to_string(index=False))
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