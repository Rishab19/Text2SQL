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
    Downloads the database_schemas.yaml file from the FINCH Hugging Face repo,
    accurately parses its top-level schema blocks, maps structural relationships,
    and returns a clean, non-collapsed DataFrame with complete metrics.
    """
    # 1. Download file from Hugging Face
    print(f"Downloading '{filename}' from repo '{repo_id}'...")
    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        repo_type="dataset"
    )

    # 2. Parse the YAML file safely
    print("Parsing YAML data...")
    with open(local_path, 'r', encoding='utf-8') as f:
        schema_data = yaml.safe_load(f)

    # If the YAML parser reads the top layer as a single dictionary wrap, open it
    if isinstance(schema_data, dict) and not any(k in schema_data for k in ['db_id', 'table_names_original']):
        # If it's formatted as a dictionary of databases
        db_entries = [{"db_id": k, **v} if isinstance(v, dict) else {"db_id": k} for k, v in schema_data.items()]
    elif isinstance(schema_data, list):
        # If it's a list of database specifications
        db_entries = schema_data
    else:
        # Fallback to literal entry check
        db_entries = [schema_data]

    # 3. Flatten the schema rows without collapsing shared cross-db names
    rows = []
    
    for entry in db_entries:
        if not isinstance(entry, dict):
            continue
            
        # Extract the true database identifier
        db_id = entry.get("db_id") or entry.get("database")
        if not db_id:
            continue
            
        # Look for explicit table metadata lists (Standard BIRD/FINCH schema specification)
        table_list = entry.get("table_names_original") or entry.get("table_names")
        column_list = entry.get("column_names_original") or entry.get("column_names")
        
        if table_list and column_list:
            # Reconstruct table structural context from standard parallel arrays
            for col_idx, col_data in enumerate(column_list):
                # Standard BIRD format: column_list elements are [table_index, column_name]
                if isinstance(col_data, list) and len(col_data) >= 2:
                    t_idx, col_name = col_data[0], col_data[1]
                    # Skip database wildcard indexes (-1)
                    if t_idx == -1: 
                        continue
                    table_name = table_list[t_idx] if t_idx < len(table_list) else "unknown"
                else:
                    table_name = "unknown"
                    col_name = str(col_data)
                    
                rows.append({
                    "database": db_id,
                    "table": table_name,
                    "column_name": col_name
                })
        else:
            # Fallback for alternative structural nesting layouts
            for key, val in entry.items():
                if key in ["db_id", "database", "table_names", "table_names_original", "column_names", "column_names_original"]:
                    continue
                if isinstance(val, dict):
                    for table_name, table_body in val.items():
                        if isinstance(table_body, dict) and "columns_info" in table_body:
                            for col in table_body["columns_info"]:
                                rows.append({
                                    "database": db_id,
                                    "table": table_name,
                                    "column_name": col.get("column_name")
                                })
                        else:
                            rows.append({
                                "database": db_id,
                                "table": table_name,
                                "column_name": None
                            })

    # 4. Construct accurate DataFrame
    df_schema = pd.DataFrame(rows)

    # 5. Extract strict metrics using full database -> table context paths
    num_databases = df_schema["database"].nunique()
    
    # Crucial Fix: Use full lineage path to avoid collapsing separate tables sharing names
    unique_tables = df_schema[["database", "table"]].drop_duplicates().shape[0]
    
    # Calculate column counts per individual table instance
    cols_per_table = (
        df_schema[df_schema["column_name"].notna()]
        .groupby(["database", "table"])["column_name"]
        .count()
        .reset_index(name="column_count")
    )

    # 6. Structured Terminal Summary Output
    print("\n" + "="*50)
    print("                FINCH SCHEMA METRICS")
    print("="*50)
    print(f"📊 TOTAL DATABASES:       {num_databases}")
    print(f"🏢 TABLES:                {unique_tables} across all databases")
    print("="*50)
    print("\n🔢 COLUMNS PER TABLE (Preview of first 10 rows):")
    print("-" * 50)
    print(cols_per_table.head(10).to_string(index=False))
    print("="*50)

    return df_schema


if __name__ == "__main__":
    token = UserSecretsClient().get_secret('HF_TOKEN')
    login(token=token.strip())
    total_remote_files = count_remote_sqlite_files(verbose=True)
    print(f"Total SQLite files found in remote repo: {total_remote_files}")
    df = summarize_hf_yaml_schema(
        repo_id="domyn/FINCH", 
        filename="schemas/database_schemas.yaml"
    )