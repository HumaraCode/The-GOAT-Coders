import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer")

# --- SIDEBAR: THEME SELECTOR ---
st.sidebar.header("🎨 Appearance")
theme_choice = st.sidebar.radio("Select Map Theme", ["Light Mode", "Dark Mode"])

# Map tile configuration based on selected theme
if theme_choice == "Dark Mode":
    map_tiles = "cartodbdark_matter"
else:
    map_tiles = "OpenStreetMap"

st.title("📍 GRIDPOINT: Warehouse Location Optimization Platform")
st.markdown("Optimize warehouse placements and minimize weighted delivery costs.")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Model Parameters")
num_warehouses = st.sidebar.slider("Number of Warehouses (K)", min_value=1, max_value=5, value=2)
enable_capacity = st.sidebar.checkbox("Enable Warehouse Capacity Limit")
capacity_limit = st.sidebar.number_input("Max Orders per Warehouse", value=1000, step=100) if enable_capacity else None

enable_radius = st.sidebar.checkbox("Enable Max Delivery Radius (km)")
max_radius_km = st.sidebar.slider("Max Service Radius (km)", 5, 50, 20) if enable_radius else None

cost_per_km = st.sidebar.number_input("Cost per Order-km ($)", value=0.5, step=0.1)

# --- DATA INPUT SECTION ---
st.sidebar.header("2. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Neighborhood CSV", type=["csv"])

# Default Sample Real-world Data (Bangalore Area Coordinates)
default_data = pd.DataFrame({
    'neighborhood': ['Koramangala', 'Indiranagar', 'Jayanagar', 'Whitefield', 'HSR Layout', 'Electronic City', 'Malleswaram'],
    'lat': [12.9352, 12.9784, 12.9250, 12.9698, 12.9121, 12.8452, 13.0031],
    'lon': [77.6245, 77.6408, 77.5938, 77.7500, 77.6446, 77.6602, 77.5643],
    'daily_orders': [450, 380, 300, 600, 500, 400, 250]
})

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = default_data

st.subheader("Neighborhood Demand Data")
st.dataframe(df, use_container_width=True)

# --- HELPER: HAVERSINE DISTANCE MATRIX (KM) ---
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- OPTIMIZATION ALGORITHM (ILP) ---
def solve_warehouse_placement(df, k, cap=None, max_rad=None):
    n = len(df)
    coords = df[['lat', 'lon']].values
    orders = df['daily_orders'].values
    
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

# --- RUN OPTIMIZATION ---
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

    # --- METRICS & COMPARISON ---
    st.subheader("📊 Optimization & Cost Comparison")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Delivery Cost", f"${orig_cost:,.2f}")
    col2.metric("Optimized Delivery Cost", f"${opt_total_cost:,.2f}")
    col3.metric("Total Cost Saved", f"${orig_cost - opt_total_cost:,.2f}", delta=f"{((orig_cost - opt_total_cost)/orig_cost)*100:.1f}%")
    col4.metric("Avg Distance / Order", f"{df['dist_to_wh_km'].mean():.2f} km")

    # --- MAP VISUALIZATION ---
    st.subheader("🗺️ Geographic Map & Route Assignments")
    map_center = [df['lat'].mean(), df['lon'].mean()]
    
    # Applied dynamic map_tiles variable here
    m = folium.Map(location=map_center, zoom_start=11, tiles=map_tiles)

    colors = ['cyan' if theme_choice == "Dark Mode" else 'red', 
              'magenta' if theme_choice == "Dark Mode" else 'blue', 
              'lime' if theme_choice == "Dark Mode" else 'green', 
              'purple', 'orange', 'yellow']

    for idx, wh_id in enumerate(wh_indices):
        wh_row = df.iloc[wh_id]
        color = colors[idx % len(colors)]
        
        folium.Marker(
            location=[wh_row['lat'], wh_row['lon']],
            popup=f"<b>WAREHOUSE: {wh_row['neighborhood']}</b>",
            icon=folium.Icon(color='red' if theme_choice == "Light Mode" else 'black', icon="building", prefix="fa")
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
            fill_opacity=0.8
        ).add_to(m)

        folium.PolyLine(
            locations=[[row['lat'], row['lon']], [wh_lat, wh_lon]],
            color=color,
            weight=2,
            opacity=0.7
        ).add_to(m)

    st_folium(m, width=1200, height=500)

except Exception as e:
    st.error("Optimization failed with current constraints. Try increasing warehouse capacity or max radius limits.")
