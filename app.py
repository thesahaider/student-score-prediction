
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sales Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Sales Data Analysis Dashboard")
st.write("Analyze sales revenue, products, and categories.")

# Load and prepare the data
df = pd.read_csv("sales.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["Month"] = df["Date"].dt.strftime("%Y-%m")

# Sidebar filters
st.sidebar.header("Filters")

categories = st.sidebar.multiselect(
    "Select categories",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

products = st.sidebar.multiselect(
    "Select products",
    options=sorted(df["Product"].unique()),
    default=sorted(df["Product"].unique())
)

filtered_df = df[
    df["Category"].isin(categories)
    & df["Product"].isin(products)
]

# Key performance indicators
total_revenue = filtered_df["Revenue"].sum()
total_units = filtered_df["Quantity"].sum()
total_orders = len(filtered_df)

top_product = (
    filtered_df.groupby("Product")["Revenue"].sum().idxmax()
    if not filtered_df.empty
    else "No data"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"Rs. {total_revenue:,.0f}")
col2.metric("Units Sold", f"{total_units:,}")
col3.metric("Sales Records", f"{total_orders:,}")
col4.metric("Top Product", top_product)

st.divider()

# Revenue by product
st.subheader("Revenue by Product")

if not filtered_df.empty:
    product_sales = (
        filtered_df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(product_sales)

    # Revenue by month
    st.subheader("Monthly Revenue")

    monthly_sales = (
        filtered_df.groupby("Month")["Revenue"]
        .sum()
        .sort_index()
    )

    st.line_chart(monthly_sales)

    # Category performance
    st.subheader("Revenue by Category")

    category_sales = (
        filtered_df.groupby("Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()
    category_sales.plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue (PKR)")
    ax.set_title("Category Revenue Comparison")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Detailed sales table
    st.subheader("Sales Records")
    st.dataframe(
        filtered_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No sales records match your selected filters.")
