import random
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select

from app.database.connection import Base, SessionLocal, engine
from app.database.models import Customer, Order, OrderItem, Product


FIRST_NAMES = [
    "Rahul", "Ananya", "Arjun", "Priya", "Kiran",
    "Sneha", "Vikram", "Aisha", "Rohan", "Divya",
    "Aditya", "Meera", "Sai", "Neha", "Varun",
]

LAST_NAMES = [
    "Sharma", "Reddy", "Patel", "Kumar", "Rao",
    "Singh", "Verma", "Gupta", "Mehta", "Naidu",
]

CITIES = [
    "Hyderabad", "Bengaluru", "Mumbai", "Chennai",
    "Pune", "Delhi", "Kolkata", "Jaipur",
    "Ahmedabad", "Kochi",
]

PRODUCT_DATA = [
    ("Wireless Mouse", "Accessories", Decimal("899.00")),
    ("Mechanical Keyboard", "Accessories", Decimal("2499.00")),
    ("USB-C Hub", "Accessories", Decimal("1799.00")),
    ("Laptop Stand", "Accessories", Decimal("1299.00")),
    ("Webcam", "Electronics", Decimal("3299.00")),
    ("Headphones", "Electronics", Decimal("3999.00")),
    ("Bluetooth Speaker", "Electronics", Decimal("2199.00")),
    ("Monitor", "Electronics", Decimal("12999.00")),
    ("SSD 1TB", "Storage", Decimal("6499.00")),
    ("External Hard Drive", "Storage", Decimal("4999.00")),
    ("Notebook Set", "Stationery", Decimal("299.00")),
    ("Office Chair", "Furniture", Decimal("8999.00")),
    ("Desk Lamp", "Furniture", Decimal("1499.00")),
    ("Power Bank", "Electronics", Decimal("1999.00")),
    ("Smart Watch", "Wearables", Decimal("5999.00")),
    ("Phone Case", "Accessories", Decimal("499.00")),
    ("Tablet", "Electronics", Decimal("18999.00")),
    ("Laptop Backpack", "Accessories", Decimal("1599.00")),
    ("Wireless Earbuds", "Electronics", Decimal("2799.00")),
    ("Ethernet Adapter", "Accessories", Decimal("799.00")),
    ("HDMI Cable", "Accessories", Decimal("399.00")),
    ("Graphics Tablet", "Electronics", Decimal("7499.00")),
    ("Portable Monitor", "Electronics", Decimal("10999.00")),
    ("Webcam Light", "Accessories", Decimal("999.00")),
    ("Microphone", "Electronics", Decimal("4499.00")),
]

STATUSES = ["delivered", "shipped", "pending", "cancelled"]
STATUS_WEIGHTS = [65, 15, 12, 8]


def seed_database() -> None:
    """Create the schema and insert reproducible sample data."""

    random.seed(42)

    # Create missing tables without replacing existing tables.
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        customer_count = session.scalar(
            select(func.count()).select_from(Customer)
        )

        # Prevent accidental duplicate seeding.
        if customer_count:
            print("Database already contains customers.")
            print("Skipping seeding to avoid duplicate records.")
            return

        customers = [
            Customer(
                name=f"{random.choice(FIRST_NAMES)} "
                     f"{random.choice(LAST_NAMES)}",
                city=random.choice(CITIES),
                email=f"customer{i + 1}@example.com",
            )
            for i in range(100)
        ]

        products = [
            Product(
                product_name=name,
                category=category,
                price=price,
            )
            for name, category, price in PRODUCT_DATA
        ]

        session.add_all(customers + products)
        session.flush()

        # All dates are within the period starting January 1, 2025.
        start_date = date(2025, 1, 1)
        orders = []

        for _ in range(500):
            order = Order(
                customer=random.choice(customers),
                order_date=start_date
                + timedelta(days=random.randint(0, 650)),
                status=random.choices(
                    STATUSES,
                    weights=STATUS_WEIGHTS,
                    k=1,
                )[0],
            )

            # Each order contains 1–4 distinct products.
            selected_products = random.sample(
                products,
                k=random.randint(1, 4),
            )

            for product in selected_products:
                order.items.append(
                    OrderItem(
                        product=product,
                        quantity=random.randint(1, 5),
                        unit_price=product.price,
                    )
                )

            orders.append(order)

        # SQLAlchemy persists associated order items through
        # the parent Order relationship.
        session.add_all(orders)
        session.commit()

        # Report actual database counts.
        counts = {
            "customers": session.scalar(
                select(func.count()).select_from(Customer)
            ),
            "products": session.scalar(
                select(func.count()).select_from(Product)
            ),
            "orders": session.scalar(
                select(func.count()).select_from(Order)
            ),
            "order_items": session.scalar(
                select(func.count()).select_from(OrderItem)
            ),
        }

        print("Database seeded successfully!")
        for table, count in counts.items():
            print(f"{table}: {count}")


if __name__ == "__main__":
    seed_database()