from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    city: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(150), unique=True, nullable=False
    )

    orders: Mapped[list["Order"]] = relationship(
        back_populates="customer"
    )


class Product(Base):
    __tablename__ = "products"

    __table_args__ = (
        CheckConstraint("price >= 0", name="ck_products_price"),
    )

    product_id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    product_name: Mapped[str] = mapped_column(
        String(150), nullable=False
    )
    category: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False
    )

    order_items: Mapped[list["OrderItem"]] = relationship(
        back_populates="product"
    )


class Order(Base):
    __tablename__ = "orders"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'shipped', 'delivered', 'cancelled')",
            name="ck_orders_status",
        ),
    )

    order_id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="RESTRICT"),
        nullable=False,
    )
    order_date: Mapped[date] = mapped_column(
        Date, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False
    )

    customer: Mapped["Customer"] = relationship(
        back_populates="orders"
    )
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_order_items_quantity",
        ),
        CheckConstraint(
            "unit_price >= 0",
            name="ck_order_items_unit_price",
        ),
        UniqueConstraint(
            "order_id",
            "product_id",
            name="uq_order_items_order_product",
        ),
    )

    order_item_id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.order_id", ondelete="RESTRICT"),
        nullable=False,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.product_id", ondelete="RESTRICT"),
        nullable=False,
    )
    quantity: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    unit_price: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False
    )

    order: Mapped["Order"] = relationship(
        back_populates="items"
    )
    product: Mapped["Product"] = relationship(
        back_populates="order_items"
    )