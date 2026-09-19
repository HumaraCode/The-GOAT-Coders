
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer & Logistics Analytics")

st.title("📍 GRIDPOINT: Warehouse Location Optimization & Logistics Analytics")
st.markdown("Optimize warehouse placements, track 3PL partner performance, and analyze historical delivery analytics.")

# --- DATA CLEANING & HELPER FUNCTIONS ---
@st.cache_data
def load_and_clean_logistics_data():
    try:
        data = pd.read_csv('Delivery_Logistics.csv')
        # Extract numerical hours from timestamp strings
        data['actual_hours'] = data['delivery_time_hours'].str.extract(r'(\d+)$').astype(float)
        data['expected_hours'] = data['expected_time_hours'].str.extract(r'(\d+)$').astype(float)
        return data
    except Exception as e:
        st.error(f"Error loading Delivery_Logistics.csv: {e}")
        return None

logistics_df = load_and_clean_logistics_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Optimization Parameters")
num_warehouses = st.sidebar.slider("Number of Warehouses (K)", min_value=1, max_value=5, value=2)
enable_capacity = st.sidebar.checkbox("Enable Warehouse Capacity Limit")
capacity_limit = st.sidebar.number_input("Max Orders per Warehouse", value=1000, step=100) if enable_capacity else None

enable_radius = st.sidebar.checkbox("Enable Max Delivery Radius (km)")
max_radius_km = st.sidebar.slider("Max Service Radius (km)", 5, 50, 20) if enable_radius else None

cost_per_km = st.sidebar.number_input("Cost per Order-km ($)", value=0.5, step=0.1)

st.sidebar.header("2. Optimization Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Neighborhood CSV", type=["csv"])

# Default Sample Real-world Data (Bangalore Area Coordinates)
default_data = pd.DataFrame({
    'neighborhood': ['Koramangala', 'Indiranagar', 'Jayanagar', 'Whitefield', 'HSR Layout', 'Electronic City', 'Malleswaram'],
    'lat': [12.9352, 12.9784, 12.9250, 12.9698, 12.9121, 12.8452, 13.0031],
    'lon': [77.6245, 77.6408, 77.5938, 77.7500, 77.6446, 77.6602, 77.5643],
    'daily_orders': [450, 380, 300, 600, 500, 400, 250]
})

df = pd.read_csv(uploaded_file) if uploaded_file is not None else default_data

# --- HAVERSINE DISTANCE MATRIX (KM) ---
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- OPTIMIZATION ALGORITHM ---
def solve_warehouse_placement(df_nodes, k, cap=None, max_rad=None):
    n = len(df_nodes)
    coords = df_nodes[['lat', 'lon']].values
    orders = df_nodes['daily_orders'].values
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = haversine_np(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])

    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("Warehouse", range(n), cat='Binary')
    x = pulp.LpVariable.dicts("Assign", (range(n), range(n)), cat='Binary')

    prob += pulp.lpSum(orders[i] * dist_matrix[i, j] * x[i][j] for i in range(n) for j in range(n))
    prob += pulp.lpSum(y[j] for j in range(n)) == k

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n)) == 1

    for i in range(n):
        for j in range(n):
            prob += x[i][j] <= y[j]

    if cap is not None:
        for j in range(n):
            prob += pulp.lpSum(orders[i] * x[i][j] for i in range(n)) <= cap * y[j]

    if max_rad is not None:
        for i in range(n):
            for j in range(n):
                if dist_matrix[i, j] > max_rad:
                    prob += x[i][j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_wh = [j for j in range(n) if pulp.value(y[j]) == 1]
    assignments = []
    for i in range(n):
        for j in selected_wh:
            if pulp.value(x[i][j]) == 1:
                assignments.append(j)
                break

    return selected_wh, assignments, dist_matrix

# --- TABBED LAYOUT ---
tab1, tab2 = st.tabs(["🏗️ Warehouse Optimization Solver", "📊 Historical Logistics Dashboard"])

# ==========================================
# TAB 1: FACILITY LOCATION SOLVER
# ==========================================
with tab1:
    st.subheader("Neighborhood Demand Data")
    st.dataframe(df, use_container_width=True)

    try:
        wh_indices, assignments, dist_matrix = solve_warehouse_placement(df, num_warehouses, capacity_limit, max_radius_km)
        df['assigned_wh_id'] = assignments
        df['assigned_wh_name'] = [df['neighborhood'].iloc[a] for a in assignments]
        df['dist_to_wh_km'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]

        opt_total_dist = (df['dist_to_wh_km'] * df['daily_orders']).sum()
        opt_total_cost = opt_total_dist * cost_per_km

        central_idx = np.argmin(dist_matrix.sum(axis=1))
        orig_dist = sum(dist_matrix[i, central_idx] * df['daily_orders'].iloc[i] for i in range(len(df)))
        orig_cost = orig_dist * cost_per_km

        st.subheader("📊 Optimization & Cost Comparison")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Delivery Cost", f"${orig_cost:,.2f}")
        col2.metric("Optimized Delivery Cost", f"${opt_total_cost:,.2f}")
        col3.metric("Total Cost Saved", f"${orig_cost - opt_total_cost:,.2f}", delta=f"{((orig_cost - opt_total_cost)/orig_cost)*100:.1f}%")
        col4.metric("Avg Distance / Order", f"{df['dist_to_wh_km'].mean():.2f} km")

        st.subheader("🗺️ Geographic Map & Route Assignments")
        map_center = [df['lat'].mean(), df['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=11)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']

        for idx, wh_id in enumerate(wh_indices):
            wh_row = df.iloc[wh_id]
            color = colors[idx % len(colors)]
            folium.Marker(
                location=[wh_row['lat'], wh_row['lon']],
                popup=f"<b>WAREHOUSE: {wh_row['neighborhood']}</b>",
                icon=folium.Icon(color=color, icon="building", prefix="fa")
            ).add_to(m)

        for i, row in df.iterrows():
            assigned_wh_idx = row['assigned_wh_id']
            color_idx = wh_indices.index(assigned_wh_idx)
            color = colors[color_idx % len(colors)]
            wh_lat = df.iloc[assigned_wh_idx]['lat']
            wh_lon = df.iloc[assigned_wh_idx]['lon']

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6 + (row['daily_orders'] / 100),
                popup=f"{row['neighborhood']}<br>Orders: {row['daily_orders']}<br>Distance: {row['dist_to_wh_km']:.2f} km",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

            folium.PolyLine(
                locations=[[row['lat'], row['lon']], [wh_lat, wh_lon]],
                color=color,
                weight=1.5,
                opacity=0.6
            ).add_to(m)

        st_folium(m, width=1200, height=500)

    except Exception as e:
        st.error(f"Optimization failed with current constraints: {e}. Try increasing warehouse capacity or max radius limits.")

# ==========================================
# TAB 2: HISTORICAL LOGISTICS ANALYTICS
# ==========================================
with tab2:
    if logistics_df is not None:
        st.subheader("📈 Historical Delivery Operations & Performance Overview")

        # Top Metric Cards
        total_shipments = len(logistics_df)
        delivered_count = (logistics_df['delivery_status'] == 'delivered').sum()
        delayed_count = (logistics_df['delayed'] == 'yes').sum()
        failed_count = (logistics_df['delivery_status'] == 'failed').sum()
        avg_cost = logistics_df['delivery_cost'].mean()
        avg_rating = logistics_df['delivery_rating'].mean()

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Total Shipments", f"{total_shipments:,}")
        mcol2.metric("On-Time Rate", f"{((delivered_count - (delayed_count - failed_count))/total_shipments)*100:.1f}%")
        mcol3.metric("Delay Rate", f"{(delayed_count/total_shipments)*100:.1f}%")
        mcol4.metric("Avg Shipment Cost", f"${avg_cost:.2f}")
        mcol5.metric("Avg CSAT Rating", f"{avg_rating:.2f} / 5.0")

        st.markdown("---")

        # 4-Grid Matplotlib/Seaborn Charts
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Chart 1: Status Distribution
        sns.countplot(data=logistics_df, x='delivery_status', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Delivery Status Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Status')
        axes[0, 0].set_ylabel('Count')
        for p in axes[0, 0].patches:
            axes[0, 0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9)

        # Chart 2: Weather vs Delay
        weather_delay = logistics_df.groupby('weather_condition')['delayed'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
        sns.barplot(data=weather_delay, x='weather_condition', y='delayed', ax=axes[0, 1], palette='magma')
        axes[0, 1].set_title('Delay Rate (%) by Weather Condition', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Weather Condition')
        axes[0, 1].set_ylabel('Delay Rate (%)')
        for p in axes[0, 1].patches:
            axes[0, 1].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9)

        # Chart 3: Partner Delay Rate
        partner_stats = logistics_df.groupby('delivery_partner').agg(
            delay_rate=('delayed', lambda x: (x == 'yes').mean() * 100)
        ).reset_index().sort_values('delay_rate', ascending=False)
        sns.barplot(data=partner_stats, x='delay_rate', y='delivery_partner', ax=axes[1, 0], palette='mako')
        axes[1, 0].set_title('Delay Rate (%) by Delivery Partner', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Delay Rate (%)')
        axes[1, 0].set_ylabel('Partner')
        for p in axes[1, 0].patches:
            axes[1, 0].annotate(f'{p.get_width():.1f}%', (p.get_width(), p.get_y() + p.get_height() / 2.),
                                ha='center', va='center', xytext=(12, 0), textcoords='offset points', fontsize=8)

        # Chart 4: Cost vs Distance
        sns.scatterplot(data=logistics_df.sample(1500, random_state=42), x='distance_km', y='delivery_cost',
                        hue='vehicle_type', alpha=0.6, ax=axes[1, 1], palette='tab10')
        axes[1, 1].set_title('Delivery Cost vs Distance (Sampled)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Distance (km)')
        axes[1, 1].set_ylabel('Delivery Cost ($)')

        plt.tight_layout()
        st.pyplot(fig)

        # Partner Breakdown Data Table
        st.subheader("📋 3PL Partner Performance Breakdown")
        partner_summary = logistics_df.groupby('delivery_partner').agg(
            total_deliveries=('delivery_id', 'count'),
            delayed_count=('delayed', lambda x: (x == 'yes').sum()),
            failed_count=('delivery_status', lambda x: (x == 'failed').sum()),
            avg_rating=('delivery_rating', 'mean'),
            avg_cost=('delivery_cost', 'mean'),
            avg_distance_km=('distance_km', 'mean')
        ).reset_index()

        partner_summary['delay_rate_%'] = (partner_summary['delayed_count'] / partner_summary['total_deliveries']) * 100
        partner_summary['failure_rate_%'] = (partner_summary['failed_count'] / partner_summary['total_deliveries']) * 100

        st.dataframe(
            partner_summary.sort_values(by='delay_rate_%', ascending=False),
            use_container_width=True
        )
    else:
        st.warning("`Delivery_Logistics.csv` could not be loaded. Please ensure the file is present in your working directory.")
      
