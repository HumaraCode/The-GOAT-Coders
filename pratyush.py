import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer", page_icon="📍")

st.title("📍 GRIDPOINT: Warehouse Location Optimization Platform")
st.markdown("Optimize warehouse placements, minimize weighted delivery costs, and balance order fulfillment capacity.")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Model Parameters")

# Let the user choose between a fixed number of warehouses or letting the model decide based on costs
optimization_mode = st.sidebar.radio("Optimization Mode", ["Fixed Number of Warehouses", "Cost-Based Auto-Selection"])

if optimization_mode == "Fixed Number of Warehouses":
    max_warehouses = st.sidebar.slider("Number of Warehouses (K)", min_value=1, max_value=10, value=2)
    exact_k = True
else:
    max_warehouses = st.sidebar.slider("Maximum Allowable Warehouses", min_value=1, max_value=10, value=5)
    exact_k = False

st.sidebar.divider()

cost_per_km = st.sidebar.number_input("Transport Cost per Order-km (₹)", value=10.0, step=1.0, help="Cost to transport one order over 1 km.")
setup_cost = st.sidebar.number_input("Warehouse Setup/Operating Cost (₹)", value=50000.0, step=5000.0, help="Fixed cost to open a warehouse. Crucial for Cost-Based Auto-Selection.")

st.sidebar.divider()

enable_capacity = st.sidebar.checkbox("Enable Warehouse Capacity Limit", value=True)
capacity_limit = st.sidebar.number_input("Max Orders per Warehouse", value=1500, step=100) if enable_capacity else None

enable_radius = st.sidebar.checkbox("Enable Max Delivery Radius (km)")
max_radius_km = st.sidebar.slider("Max Service Radius (km)", 5, 100, 20) if enable_radius else None

# --- DATA INPUT SECTION ---
st.sidebar.header("2. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Neighborhood CSV", type=["csv"], help="Columns required: neighborhood, lat, lon, daily_orders")

# Default Sample Real-world Data (Bangalore Area Coordinates)
default_data = pd.DataFrame({
    'neighborhood': ['Koramangala', 'Indiranagar', 'Jayanagar', 'Whitefield', 'HSR Layout', 'Electronic City', 'Malleswaram', 'Marathahalli', 'Yelahanka', 'Banashankari'],
    'lat': [12.9352, 12.9784, 12.9250, 12.9698, 12.9121, 12.8452, 13.0031, 12.9569, 13.1007, 12.9152],
    'lon': [77.6245, 77.6408, 77.5938, 77.7500, 77.6446, 77.6602, 77.5643, 77.7011, 77.5963, 77.5736],
    'daily_orders': [450, 380, 300, 600, 500, 400, 250, 550, 200, 350]
})

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = default_data

with st.expander("View Neighborhood Demand Data"):
    st.dataframe(df, use_container_width=True)

# --- HELPER: VECTORIZED HAVERSINE DISTANCE MATRIX (KM) ---
def compute_distance_matrix(coords):
    """Highly efficient vectorized haversine distance calculation."""
    lats = np.radians(coords[:, 0])
    lons = np.radians(coords[:, 1])
    dlat = lats[:, np.newaxis] - lats
    dlon = lons[:, np.newaxis] - lons
    a = np.sin(dlat / 2.0)**2 + np.cos(lats[:, np.newaxis]) * np.cos(lats) * np.sin(dlon / 2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

# --- OPTIMIZATION ALGORITHM (ILP) ---
@st.cache_data
def solve_warehouse_placement(df, max_k, exact_k, cap, max_rad, transport_cost, setup_cost):
    n = len(df)
    coords = df[['lat', 'lon']].values
    orders = df['daily_orders'].values
    
    # Efficient Matrix Calculation
    dist_matrix = compute_distance_matrix(coords)

    # PuLP Optimization Problem
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    
    # Variables
    y = pulp.LpVariable.dicts("Warehouse", range(n), cat='Binary')
    x = pulp.LpVariable.dicts("Assign", (range(n), range(n)), cat='Binary')

    # Objective: Minimize Total Delivery Cost + Fixed Warehouse Setup Costs
    delivery_cost = pulp.lpSum(orders[i] * dist_matrix[i, j] * transport_cost * x[i][j] for i in range(n) for j in range(n))
    facility_cost = pulp.lpSum(setup_cost * y[j] for j in range(n))
    prob += delivery_cost + facility_cost

    # Constraints:
    # 1. Warehouse limit
    if exact_k:
        prob += pulp.lpSum(y[j] for j in range(n)) == max_k
    else:
        prob += pulp.lpSum(y[j] for j in range(n)) <= max_k
        prob += pulp.lpSum(y[j] for j in range(n)) >= 1

    # 2. Every neighborhood must be assigned to exactly 1 warehouse
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n)) == 1

    # 3. Neighborhood i can only be served by j if j is a warehouse
    for i in range(n):
        for j in range(n):
            prob += x[i][j] <= y[j]

    # 4. Warehouse Capacity Limit
    if cap is not None:
        for j in range(n):
            prob += pulp.lpSum(orders[i] * x[i][j] for i in range(n)) <= cap * y[j]

    # 5. Max Service Radius Limit
    if max_rad is not None:
        for i in range(n):
            for j in range(n):
                if dist_matrix[i, j] > max_rad:
                    prob += x[i][j] == 0

    # Solve quietly
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None, None, False

    # Extract Results
    selected_wh = [j for j in range(n) if pulp.value(y[j]) == 1]
    assignments = []
    for i in range(n):
        for j in selected_wh:
            if pulp.value(x[i][j]) == 1:
                assignments.append(j)
                break

    return selected_wh, assignments, dist_matrix, True

# --- RUN OPTIMIZATION & RENDER UI ---
wh_indices, assignments, dist_matrix, is_optimal = solve_warehouse_placement(
    df, max_warehouses, exact_k, capacity_limit, max_radius_km, cost_per_km, setup_cost
)

if not is_optimal:
    st.error("⚠️ Optimization failed. No feasible solution found. Try increasing warehouse capacity, the number of warehouses, or the max delivery radius.")
else:
    df['assigned_wh_id'] = assignments
    df['assigned_wh_name'] = [df['neighborhood'].iloc[a] for a in assignments]
    df['dist_to_wh_km'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]
    df['delivery_cost_inr'] = df['dist_to_wh_km'] * df['daily_orders'] * cost_per_km

    # Metrics Calculations
    opt_total_delivery = df['delivery_cost_inr'].sum()
    opt_setup_cost = len(wh_indices) * setup_cost
    opt_total_cost = opt_total_delivery + opt_setup_cost

    # Baseline Comparison: Centralized Single Hub (Best single location)
    central_idx = np.argmin(dist_matrix.sum(axis=1))
    orig_delivery = sum(dist_matrix[i, central_idx] * df['daily_orders'].iloc[i] * cost_per_km for i in range(len(df)))
    orig_total_cost = orig_delivery + setup_cost # Setup 1 warehouse

    # Tabs for better UI organization
    tab1, tab2, tab3 = st.tabs(["🗺️ Map Visualization", "📊 Financial & Network Stats", "📋 Detailed Data"])

    with tab1:
        st.subheader("Geographic Map & Route Assignments")
        map_center = [df['lat'].mean(), df['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=11)

        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

        # Draw Warehouses
        for idx, wh_id in enumerate(wh_indices):
            wh_row = df.iloc[wh_id]
            color = colors[idx % len(colors)]
            folium.Marker(
                location=[wh_row['lat'], wh_row['lon']],
                popup=f"<b>HUB: {wh_row['neighborhood']}</b>",
                icon=folium.Icon(color="black", icon_color=color, icon="building", prefix="fa")
            ).add_to(m)

        # Draw Neighborhoods and Connective Lines
        for i, row in df.iterrows():
            assigned_wh_idx = row['assigned_wh_id']
            color_idx = wh_indices.index(assigned_wh_idx)
            color = colors[color_idx % len(colors)]
            
            wh_lat = df.iloc[assigned_wh_idx]['lat']
            wh_lon = df.iloc[assigned_wh_idx]['lon']

            # Delivery Nodes
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6 + (row['daily_orders'] / 150),
                popup=f"<b>{row['neighborhood']}</b><br>Orders: {row['daily_orders']}<br>Cost: ₹{row['delivery_cost_inr']:.2f}",
                color=color,
                fill=True,
                fill_opacity=0.8
            ).add_to(m)

            # Route Lines
            folium.PolyLine(
                locations=[[row['lat'], row['lon']], [wh_lat, wh_lon]],
                color=color,
                weight=2,
                opacity=0.6,
                dash_array='5, 5'
            ).add_to(m)

        st_folium(m, width=1200, height=550)

    with tab2:
        st.subheader("Cost Comparison")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Cost (1 Hub)", f"₹{orig_total_cost:,.2f}")
        col2.metric("Optimized Total Cost", f"₹{opt_total_cost:,.2f}", help="Includes setup costs and delivery costs.")
        col3.metric("Total Cost Saved", f"₹{orig_total_cost - opt_total_cost:,.2f}", delta=f"{((orig_total_cost - opt_total_cost)/orig_total_cost)*100:.1f}%")
        col4.metric("Avg Distance / Order", f"{df['dist_to_wh_km'].mean():.2f} km")

        st.subheader("Warehouse Utilization")
        # Compute warehouse stats
        wh_stats = df.groupby('assigned_wh_name').agg(
            total_orders=('daily_orders', 'sum'),
            neighborhoods_served=('neighborhood', 'count'),
            avg_delivery_dist=('dist_to_wh_km', 'mean')
        ).reset_index()
        
        if capacity_limit:
            wh_stats['utilization_%'] = (wh_stats['total_orders'] / capacity_limit) * 100
            
        st.dataframe(wh_stats.style.format(precision=2), use_container_width=True)

    with tab3:
        st.subheader("Neighborhood Assignment Breakdown")
        display_df = df[['neighborhood', 'daily_orders', 'assigned_wh_name', 'dist_to_wh_km', 'delivery_cost_inr']].copy()
        display_df.rename(columns={
            'neighborhood': 'Neighborhood',
            'daily_orders': 'Daily Orders',
            'assigned_wh_name': 'Assigned Hub',
            'dist_to_wh_km': 'Distance (km)',
            'delivery_cost_inr': 'Delivery Cost (₹)'
        }, inplace=True)
        st.dataframe(display_df.style.format(precision=2), use_container_width=True)
      
