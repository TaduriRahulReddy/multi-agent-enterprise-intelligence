from pathlib import Path
import duckdb


# --------------------------------------------------
# Database path
# --------------------------------------------------

DATABASE_PATH = Path("database/olist.duckdb")


# --------------------------------------------------
# Connect to DuckDB
# --------------------------------------------------

connection = duckdb.connect(str(DATABASE_PATH))

print("DuckDB Validation")
print("=" * 60)


# --------------------------------------------------
# 1. Check all tables
# --------------------------------------------------

print("\n1. TABLE CHECK")
print("-" * 60)

tables = connection.execute("SHOW TABLES").fetchall()

for table in tables:
    print(f"  {table[0]}")


# --------------------------------------------------
# 2. Check row counts
# --------------------------------------------------

print("\n2. ROW COUNTS")
print("-" * 60)

table_names = [
    "customers",
    "geolocation",
    "order_items",
    "payments",
    "reviews",
    "orders",
    "products",
    "sellers",
    "category_translation",
]

for table_name in table_names:
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0]

    print(f"{table_name}: {row_count:,}")


# --------------------------------------------------
# 3. Check order status distribution
# --------------------------------------------------

print("\n3. ORDER STATUS DISTRIBUTION")
print("-" * 60)

order_status_results = connection.execute(
    """
    SELECT
        order_status,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY order_status
    ORDER BY order_count DESC
    """
).fetchall()

for status, count in order_status_results:
    print(f"{status}: {count:,}")


# --------------------------------------------------
# 4. Calculate total product revenue
# --------------------------------------------------

print("\n4. TOTAL PRODUCT REVENUE")
print("-" * 60)

total_product_revenue = connection.execute(
    """
    SELECT
        SUM(price) AS total_product_revenue
    FROM order_items
    """
).fetchone()[0]

print(f"Total product revenue: {total_product_revenue:,.2f}")


# --------------------------------------------------
# 5. Calculate total freight value
# --------------------------------------------------

print("\n5. TOTAL FREIGHT VALUE")
print("-" * 60)

total_freight_value = connection.execute(
    """
    SELECT
        SUM(freight_value) AS total_freight_value
    FROM order_items
    """
).fetchone()[0]

print(f"Total freight value: {total_freight_value:,.2f}")


# --------------------------------------------------
# 6. Check average review score
# --------------------------------------------------

print("\n6. AVERAGE REVIEW SCORE")
print("-" * 60)

average_review_score = connection.execute(
    """
    SELECT
        AVG(review_score) AS average_review_score
    FROM reviews
    """
).fetchone()[0]

print(f"Average review score: {average_review_score:.2f}")


# --------------------------------------------------
# 7. Validate a multi-table join
# --------------------------------------------------

print("\n7. MULTI-TABLE JOIN TEST")
print("-" * 60)

join_test = connection.execute(
    """
    SELECT
        o.order_id,
        c.customer_unique_id,
        oi.product_id,
        p.product_category_name,
        oi.price,
        s.seller_state
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN order_items oi
        ON o.order_id = oi.order_id
    JOIN products p
        ON oi.product_id = p.product_id
    JOIN sellers s
        ON oi.seller_id = s.seller_id
    LIMIT 5
    """
).fetchall()

for row in join_test:
    print(row)


# --------------------------------------------------
# 8. Top product categories by revenue
# --------------------------------------------------

print("\n8. TOP PRODUCT CATEGORIES BY REVENUE")
print("-" * 60)

top_categories = connection.execute(
    """
    SELECT
        COALESCE(
            ct.product_category_name_english,
            p.product_category_name,
            'unknown'
        ) AS category,
        SUM(oi.price) AS revenue
    FROM order_items oi
    JOIN products p
        ON oi.product_id = p.product_id
    LEFT JOIN category_translation ct
        ON p.product_category_name = ct.product_category_name
    GROUP BY category
    ORDER BY revenue DESC
    LIMIT 10
    """
).fetchall()

for category, revenue in top_categories:
    print(f"{category}: {revenue:,.2f}")


# --------------------------------------------------
# 9. Check repeat customers
# --------------------------------------------------

print("\n9. REPEAT CUSTOMER CHECK")
print("-" * 60)

repeat_customer_stats = connection.execute(
    """
    SELECT
        COUNT(*) AS repeat_customers
    FROM (
        SELECT
            c.customer_unique_id,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM customers c
        JOIN orders o
            ON c.customer_id = o.customer_id
        GROUP BY c.customer_unique_id
        HAVING COUNT(DISTINCT o.order_id) > 1
    )
    """
).fetchone()[0]

print(f"Repeat customers: {repeat_customer_stats:,}")


# --------------------------------------------------
# 10. Close connection
# --------------------------------------------------

connection.close()

print("\nDatabase validation completed successfully.")