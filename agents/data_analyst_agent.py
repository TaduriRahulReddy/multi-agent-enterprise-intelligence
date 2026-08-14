from tools.sql_tool import execute_sql


def top_categories_by_revenue(limit=5):
    query = f"""
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
    LIMIT {limit}
    """

    return execute_sql(query)


def order_status_summary():
    query = """
    SELECT
        order_status,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY order_status
    ORDER BY order_count DESC
    """

    return execute_sql(query)


def average_review_score():
    query = """
    SELECT
        AVG(review_score) AS average_review_score
    FROM reviews
    """

    return execute_sql(query)


def repeat_customer_count():
    query = """
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

    return execute_sql(query)


def total_revenue():
    query = """
    SELECT
        SUM(price) AS total_product_revenue
    FROM order_items
    """

    return execute_sql(query)


if __name__ == "__main__":

    print("Testing Data Analyst Agent")
    print("=" * 60)

    print("\nTop 5 categories by revenue:")
    print(top_categories_by_revenue(5))

    print("\nOrder status summary:")
    print(order_status_summary())

    print("\nAverage review score:")
    print(average_review_score())

    print("\nRepeat customer count:")
    print(repeat_customer_count())

    print("\nTotal product revenue:")
    print(total_revenue())

    print("\nData Analyst Agent Version 1 test completed successfully.")