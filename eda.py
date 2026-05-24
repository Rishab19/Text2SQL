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
    Correctly steps into the 4 source benchmark wrappers (bird, spider, etc.) 
    to extract the true 33 nested databases, their 292 individual tables,
    and returns a clean summary showing total columns alongside max values.
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

    rows = []
    
    # schema_data is a list of the 4 parent benchmark blocks (bird, book_sql, bull, spider)
    for benchmark_block in schema_data:
        if not isinstance(benchmark_block, dict):
            continue
            
        # The suite name (e.g., 'bird' or 'spider')
        suite_name = benchmark_block.get("database", "unknown")
        
        # Look at the other keys at this level to find the actual database configurations
        for key, value in benchmark_block.items():
            if key == "database" or not isinstance(value, dict):
                continue
                
            # Key is the actual database profile (e.g., 'debit_card_specializing')
            actual_db_id = key
            
            # Inside the database block, extract the table dictionaries
            for table_name, table_body in value.items():
                if not isinstance(table_body, dict) or "columns_info" not in table_body:
                    # Capture empty or shell tables with 0 columns
                    rows.append({
                        "suite": suite_name,
                        "database": actual_db_id,
                        "table": table_name,
                        "column_name": None
                    })
                    continue
                
                # Extract columns
                for col in table_body["columns_info"]:
                    rows.append({
                        "suite": suite_name,
                        "database": actual_db_id,
                        "table": table_name,
                        "column_name": col.get("column_name")
                    })

    # 3. Create DataFrame
    df_schema = pd.DataFrame(rows)

    # 4. Compute perfect metrics by targeting the real database level
    num_suites = df_schema["suite"].nunique()
    num_databases = df_schema["database"].nunique()
    
    # Calculate unique tables using the database name + table name combination
    unique_tables_df = df_schema[["database", "table"]].drop_duplicates()
    num_tables = unique_tables_df.shape[0]

    # Calculate column counts per table instance
    cols_per_table = (
        df_schema[df_schema["column_name"].notna()]
        .groupby(["database", "table"])["column_name"]
        .count()
        .reset_index(name="column_count")
    )
    
    # Extract total column sums and upper limits per database
    db_column_stats = (
        cols_per_table.groupby("database")["column_count"]
        .agg(["max", "sum"])
        .reset_index()
    )
    
    # Rename columns for a clean presentation
    db_column_stats.columns = ["database", "max_columns", "total_columns"]
    
    # Sort the table by the highest absolute total column volume
    db_column_stats = db_column_stats.sort_values(by="total_columns", ascending=False)
    
    # Compute global metrics across all tables
    global_summary = pd.DataFrame([{
        "database": "GLOBAL TOTAL / ALL DBS",
        "max_columns": cols_per_table["column_count"].max(),
        "total_columns": cols_per_table["column_count"].sum()
    }])

    # 6. Display Precise Summary Output
    print("\n" + "="*55)
    print("                      FINCH SCHEMA METRICS")
    print("="*55)
    print(f"📊 TOTAL DATABASES:       {num_databases}")
    print(f"🏢 TOTAL TABLES:          {num_tables} across all databases")
    print("="*55)
    print("\n📊 COLUMN DATA SUMMARY (BY DATABASE):")
    print("-" * 55)
    print(db_column_stats.to_string(index=False))
    print("="*55)
    print("\n📊 DATASET GLOBAL SUMMARY:")
    print("-" * 55)
    print(global_summary.to_string(index=False))
    print("="*55)

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