from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/structured")


print("Olist Relationship Validation")
print("=" * 60)


# Load the core tables
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")


# --------------------------------------------------
# 1. Check primary-key uniqueness
# --------------------------------------------------

print("\n1. PRIMARY KEY UNIQUENESS")
print("-" * 60)

primary_key_checks = {
    "customers.customer_id": customers["customer_id"],
    "orders.order_id": orders["order_id"],
    "products.product_id": products["product_id"],
    "sellers.seller_id": sellers["seller_id"],
}

for name, series in primary_key_checks.items():
    total_rows = len(series)
    unique_values = series.nunique()
    duplicate_count = total_rows - unique_values

    print(f"{name}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique values: {unique_values:,}")
    print(f"  Duplicate key values: {duplicate_count:,}")
    print()


# --------------------------------------------------
# 2. Check compound key uniqueness for order items
# --------------------------------------------------

print("\n2. ORDER ITEM COMPOUND KEY")
print("-" * 60)

order_item_duplicates = order_items.duplicated(
    subset=["order_id", "order_item_id"]
).sum()

print("Key: order_id + order_item_id")
print(f"Duplicate combinations: {order_item_duplicates:,}")


# --------------------------------------------------
# 3. Foreign-key relationship checks
# --------------------------------------------------

print("\n3. FOREIGN KEY RELATIONSHIPS")
print("-" * 60)


def check_foreign_key(
    child_df,
    child_column,
    parent_df,
    parent_column,
    relationship_name,
):
    child_values = set(child_df[child_column].dropna().unique())
    parent_values = set(parent_df[parent_column].dropna().unique())

    missing_values = child_values - parent_values

    print(relationship_name)
    print(f"  Unique child keys: {len(child_values):,}")
    print(f"  Missing parent keys: {len(missing_values):,}")

    if len(missing_values) == 0:
        print("  Status: VALID")
    else:
        print("  Status: WARNING")

    print()


check_foreign_key(
    orders,
    "customer_id",
    customers,
    "customer_id",
    "orders.customer_id -> customers.customer_id",
)

check_foreign_key(
    order_items,
    "order_id",
    orders,
    "order_id",
    "order_items.order_id -> orders.order_id",
)

check_foreign_key(
    payments,
    "order_id",
    orders,
    "order_id",
    "payments.order_id -> orders.order_id",
)

check_foreign_key(
    reviews,
    "order_id",
    orders,
    "order_id",
    "reviews.order_id -> orders.order_id",
)

check_foreign_key(
    order_items,
    "product_id",
    products,
    "product_id",
    "order_items.product_id -> products.product_id",
)

check_foreign_key(
    order_items,
    "seller_id",
    sellers,
    "seller_id",
    "order_items.seller_id -> sellers.seller_id",
)


# --------------------------------------------------
# 4. Check customer_unique_id behavior
# --------------------------------------------------

print("\n4. CUSTOMER UNIQUE ID CHECK")
print("-" * 60)

print(f"Customer rows: {len(customers):,}")
print(
    f"Unique customer_id values: "
    f"{customers['customer_id'].nunique():,}"
)
print(
    f"Unique customer_unique_id values: "
    f"{customers['customer_unique_id'].nunique():,}"
)

repeat_customers = (
    customers.groupby("customer_unique_id")["customer_id"]
    .nunique()
    .gt(1)
    .sum()
)

print(
    f"Customers appearing with multiple customer_id values: "
    f"{repeat_customers:,}"
)


print("\nValidation complete.")