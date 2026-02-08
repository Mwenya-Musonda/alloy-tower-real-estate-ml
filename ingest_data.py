import os
import sys
from datetime import datetime
import logging

import pandas as pd
import snowflake.connector

# ────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────

SNOWFLAKE_CONFIG = {
    "user":     "ALLOY_TOWER_DS",
    "password": "SpecialScientist123#",
    "account":  "PXCWMVP-QF73825",
    "warehouse": "ALLOY_TOWER_WH",
    "database":  "ALLOY_TOWER_DB",
    "schema":    "SILVER",
    
}

# Table names exactly as confirmed (lowercase)
TARGET_TABLES = [
    "clean_sales_listings",
    "clean_sales_history",
    "clean_sales_agents",
    "clean_sales_offices",
]

# Where to save the downloaded files
OUTPUT_DIR = "./ml/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def get_connection():
    """Create Snowflake connection"""
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        logger.info("Successfully connected to Snowflake")
        return conn
    except Exception as e:
        logger.error("Connection failed", exc_info=True)
        sys.exit(1)


def show_real_tables(conn):
    """Try to list tables and describe one known table for debugging"""
    cur = conn.cursor()
    try:
        logger.info("Attempting to list tables in SILVER schema...")
        cur.execute("SHOW TABLES IN ALLOY_TOWER_DB.SILVER")
        rows = cur.fetchall()
        if rows:
            logger.info("Tables found:")
            for row in rows:
                name = row[1]  # table name is usually column 1
                logger.info(f"  - {name}")
        else:
            logger.warning("No tables visible in SHOW TABLES – check privileges")

        # Quick test: try to describe the main table
        logger.info("Trying to DESCRIBE clean_sales_listings...")
        cur.execute("DESCRIBE TABLE clean_sales_listings")
        desc_rows = cur.fetchall()
        if desc_rows:
            logger.info(f"clean_sales_listings exists! It has {len(desc_rows)} columns.")
        else:
            logger.info("DESCRIBE failed – table not visible/authorized")
    except Exception as e:
        logger.error(f"Cannot list or describe tables: {str(e)}")
    finally:
        cur.close()


def extract_table(conn, table_name: str) -> pd.DataFrame | None:
    """Extract one table"""
    query = f"SELECT * FROM {SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{table_name}"
    logger.info(f"Executing: {query}")

    try:
        df = pd.read_sql(query, conn)
        row_count = df.shape[0]
        col_count = df.shape[1]
        logger.info(f"Success → {table_name}: {row_count:,} rows × {col_count} columns")
        if row_count > 0:
            logger.info(f"Sample head:\n{df.head(3).to_string(index=False)}")
        return df
    except Exception as e:
        logger.error(f"Failed to extract {table_name}: {str(e)}")
        return None


def save_dataframe(df: pd.DataFrame, table_name: str):
    """Save DataFrame with timestamp"""
    if df is None or df.empty:
        logger.warning(f"No data to save for {table_name}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name}_{ts}.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────

def main():
    conn = get_connection()

    # Diagnostic: try to see what tables are visible
    show_real_tables(conn)

    # Extract each table
    extracted_count = 0
    for table in TARGET_TABLES:
        df = extract_table(conn, table)
        if df is not None:
            save_dataframe(df, table)
            extracted_count += 1

    conn.close()

    if extracted_count == len(TARGET_TABLES):
        logger.info("✅ All tables extracted and saved successfully!")
    else:
        logger.warning(f"Only {extracted_count}/{len(TARGET_TABLES)} tables were extracted.")


if __name__ == "__main__":
    main()