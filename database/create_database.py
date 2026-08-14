from pathlib import Path
import duckdb


# --------------------------------------------------
# Project paths
# --------------------------------------------------

DATA_DIR = Path("data/structured")
DATABASE_DIR = Path("database")
DATABASE_PATH = DATABASE_DIR / "olist.duckdb"


# --------------------------------------------------
# Connect to DuckDB
# --------------------------------------------------

connection = duckdb.connect(str(DATABASE_PATH))

print("Creating DuckDB database...")
print(f"Database path: {DATABASE_PATH}")
print("=" * 60)


# --------------------------------------------------
# Table definitions
# --------------------------------------------------

tables = {
    "customers": DATA_DIR / "olist_customers_dataset.csv",
    "geolocation": DATA_DIR / "olist_geolocation_dataset.csv",
    "order_items": DATA_DIR / "olist_order_items_dataset.csv",
    "payments": DATA_DIR / "olist_order_payments_dataset.csv",
    "reviews": DATA_DIR / "olist_order_reviews_dataset.csv",
    "orders": DATA_DIR / "olist_orders_dataset.csv",
    "products": DATA_DIR / "olist_products_dataset.csv",
    "sellers": DATA_DIR / "olist_sellers_dataset.csv",
    "category_translation": DATA_DIR / "product_category_name_translation.csv",
}


# --------------------------------------------------
# Load CSV files into DuckDB
# --------------------------------------------------

for table_name, csv_path in tables.items():

    print(f"Loading table: {table_name}")

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *
        FROM read_csv_auto(
            '{csv_path}',
            HEADER = TRUE
        )
        """
    )

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0]

    print(f"  Rows loaded: {row_count:,}")


# --------------------------------------------------
# Display created tables
# --------------------------------------------------

print("\nTables created:")
print("-" * 60)

created_tables = connection.execute(
    """
    SHOW TABLES
    """
).fetchall()

for table in created_tables:
    print(f"  {table[0]}")


# --------------------------------------------------
# Close database connection
# --------------------------------------------------

connection.close()

print("\nDuckDB database created successfully.")