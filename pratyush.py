
   import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer", page_icon="📍")

st.title("📍 GRIDPOINT: Warehouse Location Optimization Platform")
st.markdown("""
**How it works:** 
1. Enter the neighborhoods where your customers are placing orders (Demand Zones).
2. The AI will analyze the geographic spread and order volume. Instead of just picking one of your customer zones, it will calculate **brand new, independent coordinates** for your Optimal Warehouse(s) to make deliveries as cheap and efficient as possible.
""")

if 'locations_df' not in st.session_state:
    st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None

st.sidebar.header("1. Model Parameters")

optimization_mode = st.sidebar.radio("Optimization Mode", ["Fixed Number of Warehouses", "Cost-Based Auto-Selection"])

if optimization_mode == "Fixed Number of Warehouses":
    max_warehouses = st.sidebar.slider("Number of Warehouses to Open", min_value=1, max_value=5, value=1)
    exact_k = True
else:
    max_warehouses = st.sidebar.slider("Maximum Allowable Warehouses", min_value=1, max_value=5, value=3)
    exact_k = False

st.sidebar.divider()

cost_per_km = st.sidebar.number_input("Transport Cost per Order-km (₹)", value=15.0, step=1.0, help="Cost to transport one order over 1 km.")
setup_cost = st.sidebar.number_input("Warehouse Setup/Operating Cost (₹)", value=50000.0, step=5000.0, help="Fixed cost to open a warehouse.")

st.sidebar.divider()

enable_capacity = st.sidebar.checkbox("Enable Warehouse Capacity Limit", value=False)
capacity_limit = st.sidebar.number_input("Max Orders per Warehouse", value=1500, step=100) if enable_capacity else None

enable_radius = st.sidebar.checkbox("Enable Max Delivery Radius (km)", value=False)
max_radius_km = st.sidebar.slider("Max Service Radius (km)", 5, 100, 20) if enable_radius else None

st.header("2. Add Customer Demand Zones")
search_query = st.text_input("🔍 Search Customer Location:", placeholder="e.g. Indiranagar, Bengaluru")

if search_query:
    geolocator = Nominatim(user_agent="gridpoint_optimizer_app_v2")
    with st.spinner("Searching for locations..."):
        try:
            results = geolocator.geocode(search_query, exactly_one=False, limit=5, timeout=10)
            
            if results:
                options = {res.address: (res.latitude, res.longitude) for res in results}
                st.info("Select the correct location from the matches below:")
                selected_address = st.selectbox("Matching Results:", list(options.keys()))
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    daily_orders = st.number_input("Daily Orders from this Area", min_value=1, value=100, step=50)
                with col2:
                    st.write("") 
                    st.write("") 
                    if st.button("➕ Add Demand Zone", type="primary"):
                        short_name = selected_address.split(',')[0] 
                        lat, lon = options[selected_address]
                        
                        new_data = pd.DataFrame([{
                            'Demand Zone': short_name, 
                            'Full Address': selected_address, 
                            'Lat': lat, 
                            'Lon': lon, 
                            'Daily Orders': daily_orders
                        }])
                        
                        st.session_state.locations_df = pd.concat([st.session_state.locations_df, new_data], ignore_index=True)
                        st.session_state.opt_results = None 
                        st.rerun() 
            else:
                st.warning("No locations found. Try a different spelling or add the city name.")
        except GeocoderTimedOut:
            st.error("Location service timed out. Please try again.")
        except Exception as e:
            st.error("Error connecting to location services.")

st.subheader("Current Order Demand by Customer Zone")
if not st.session_state.locations_df.empty:
    edited_df = st.data_editor(
        st.session_state.locations_df, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True,
        disabled=["Lat", "Lon", "Full Address"] 
    )
    st.session_state.locations_df = edited_df
else:
    st.info("Your network is currently empty. Add customer locations above to begin.")

if st.button("🗑️ Clear All Locations"):
    st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
    st.session_state.opt_results = None
    st.rerun()

# --- OPTIMIZATION ALGORITHMS ---

def compute_distance_matrix(coords1, coords2):
    """Computes distance between customer coords (N) and candidate warehouse coords (M)"""
    lats1, lons1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lats2, lons2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    
    dlat = lats2 - lats1[:, np.newaxis]
    dlon = lons2 - lons1[:, np.newaxis]
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lats1[:, np.newaxis]) * np.cos(lats2) * np.sin(dlon / 2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def solve_warehouse_placement(df, max_k, exact_k, cap, max_rad, transport_cost, setup_cost):
    num_cust = len(df)
    cust_coords = df[['Lat', 'Lon']].astype(float).values
    orders = df['Daily Orders'].astype(float).values
    
    # 1. Generate Candidate Warehouse Locations (Continuous Space Grid + Center of Gravity)
    min_lat, max_lat = cust_coords[:, 0].min(), cust_coords[:, 0].max()
    min_lon, max_lon = cust_coords[:, 1].min(), cust_coords[:, 1].max()
    
    # Pad grid slightly
    lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.05
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.05
    
    # Create a 6x6 grid of candidate coordinate locations
    lat_grid = np.linspace(min_lat - lat_pad, max_lat + lat_pad, 6)
    lon_grid = np.linspace(min_lon - lon_pad, max_lon + lon_pad, 6)
    
    candidates = [[lat, lon] for lat in lat_grid for lon in lon_grid]
    
    # Add the pure Demand-Weighted Center of Gravity as a prime candidate
    total_orders = orders.sum()
    cg_lat = np.sum(cust_coords[:, 0] * orders) / total_orders
    cg_lon = np.sum(cust_coords[:, 1] * orders) / total_orders
    candidates.append([cg_lat, cg_lon])
    
    cand_coords = np.array(candidates)
    num_cand = len(cand_coords)
    
    # 2. Distance Matrix (Customers x Candidates)
    dist_matrix = compute_distance_matrix(cust_coords, cand_coords)
    
    # 3. Optimization Model
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("Warehouse", range(num_cand), cat='Binary')
    x = pulp.LpVariable.dicts("Assign", (range(num_cust), range(num_cand)), cat='Binary')

    delivery_cost = pulp.lpSum(orders[i] * dist_matrix[i, j] * transport_cost * x[i][j] for i in range(num_cust) for j in range(num_cand))
    facility_cost = pulp.lpSum(setup_cost * y[j] for j in range(num_cand))
    prob += delivery_cost + facility_cost

    if exact_k:
        prob += pulp.lpSum(y[j] for j in range(num_cand)) == max_k
    else:
        prob += pulp.lpSum(y[j] for j in range(num_cand)) <= max_k
        prob += pulp.lpSum(y[j] for j in range(num_cand)) >= 1

    for i in range(num_cust):
        prob += pulp.lpSum(x[i][j] for j in range(num_cand)) == 1

    for i in range(num_cust):
        for j in range(num_cand):
            prob += x[i][j] <= y[j]

    if cap is not None:
        for j in range(num_cand):
            prob += pulp.lpSum(orders[i] * x[i][j] for i in range(num_cust)) <= cap * y[j]

    if max_rad is not None:
        for i in range(num_cust):
            for j in range(num_cand):
                if dist_matrix[i, j] > max_rad:
                    prob += x[i][j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None, None, None, False

    # Extract Results
    selected_wh_idx = [j for j in range(num_cand) if pulp.value(y[j]) == 1]
    selected_wh_coords = cand_coords[selected_wh_idx]
    
    assignments = []
    for i in range(num_cust):
        for j in selected_wh_idx:
            if pulp.value(x[i][j]) == 1:
                assignments.append(j)
                break

    return selected_wh_idx, selected_wh_coords, assignments, dist_matrix, True

# --- EXECUTION ---
st.divider()

if st.button("🚀 Calculate Optimal Independent Warehouse Locations", type="primary", use_container_width=True):
    if len(st.session_state.locations_df) < 2:
        st.error("Please add at least TWO customer demand zones to run the optimization.")
    else:
        df_copy = st.session_state.locations_df.copy()
        
        with st.spinner("⚙️ Generating geographic grid and evaluating independent optimal coordinates..."):
            wh_idx, wh_coords, assignments, dist_matrix, is_optimal = solve_warehouse_placement(
                df_copy, max_warehouses, exact_k, capacity_limit, max_radius_km, cost_per_km, setup_cost
            )

        if not is_optimal:
            st.error("⚠️ Optimization failed. Constraints (like capacity or radius) are too strict for the given demand.")
            st.session_state.opt_results = None
        else:
            st.success("✅ Optimization Complete! AI has pinpointed the ideal coordinates for your new warehouse(s).")
            st.session_state.opt_results = {
                'df': df_copy,
                'wh_idx': wh_idx,
                'wh_coords': wh_coords,
                'assignments': assignments,
                'dist_matrix': dist_matrix
            }

# --- RENDER RESULTS ---
if st.session_state.opt_results is not None:
    res = st.session_state.opt_results
    df = res['df'].copy()
    wh_idx = res['wh_idx']
    wh_coords = res['wh_coords']
    assignments = res['assignments']
    dist_matrix = res['dist_matrix']
    
    # Map assignment to specific warehouse IDs based on the optimized coordinates
    wh_mapping = {original_j: idx for idx, original_j in enumerate(wh_idx)}
    mapped_assignments = [wh_mapping[j] for j in assignments]
    
    df['assigned_wh_id'] = [f"Warehouse {a + 1}" for a in mapped_assignments]
    df['dist_to_wh_km'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]
    df['delivery_cost_inr'] = df['dist_to_wh_km'] * df['Daily Orders'].astype(float) * cost_per_km

    opt_total_delivery = df['delivery_cost_inr'].sum()
    opt_setup_cost = len(wh_coords) * setup_cost
    opt_total_cost = opt_total_delivery + opt_setup_cost

    # Original cost calculation (using purely the weighted center of gravity as a baseline 1-hub setup)
    cust_coords = df[['Lat', 'Lon']].astype(float).values
    orders = df['Daily Orders'].astype(float).values
    cg_lat = np.sum(cust_coords[:, 0] * orders) / orders.sum()
    cg_lon = np.sum(cust_coords[:, 1] * orders) / orders.sum()
    cg_dist_matrix = compute_distance_matrix(cust_coords, np.array([[cg_lat, cg_lon]]))
    orig_delivery = sum(cg_dist_matrix[i, 0] * orders[i] * cost_per_km for i in range(len(df)))
    orig_total_cost = orig_delivery + setup_cost

    tab1, tab2, tab3 = st.tabs(["🗺️ Map Visualization", "📊 Financial & Network Stats", "📋 Detailed Routing Data"])

    with tab1:
        st.subheader("Geographic Map & Warehouse Delivery Routes")
        map_center = [df['Lat'].astype(float).mean(), df['Lon'].astype(float).mean()]
        m = folium.Map(location=map_center, zoom_start=11)

        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

        # Draw newly calculated AI Warehouses
        for idx, coord in enumerate(wh_coords):
            color = colors[idx % len(colors)]
            folium.Marker(
                location=[coord[0], coord[1]],
                popup=f"<b>🏢 NEW OPTIMAL WAREHOUSE {idx + 1}</b><br>Lat: {coord[0]:.4f}, Lon: {coord[1]:.4f}",
                icon=folium.Icon(color="black", icon_color=color, icon="building", prefix="fa")
            ).add_to(m)

        # Draw customer delivery points and connection lines
        for i, row in df.iterrows():
            a_id = mapped_assignments[i]
            color = colors[a_id % len(colors)]
            
            wh_lat = wh_coords[a_id][0]
            wh_lon = wh_coords[a_id][1]
            orders_val = float(row['Daily Orders'])

            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=6 + (orders_val / 150),
                popup=f"<b>🛒 Customer Zone: {row['Demand Zone']}</b><br>Orders: {orders_val}<br>Delivery Cost: ₹{row['delivery_cost_inr']:,.2f}",
                color=color,
                fill=True,
                fill_opacity=0.8
            ).add_to(m)

            folium.PolyLine(
                locations=[[row['Lat'], row['Lon']], [wh_lat, wh_lon]],
                color=color,
                weight=2,
                opacity=0.6,
                dash_array='5, 5'
            ).add_to(m)

        st_folium(m, width=1200, height=550)

    with tab2:
        st.subheader("Cost Comparison")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Cost (Center of Gravity)", f"₹{orig_total_cost:,.2f}")
        col2.metric("Optimized Total Cost", f"₹{opt_total_cost:,.2f}", help="Includes setup costs and delivery costs.")
        col3.metric("Total Cost Saved", f"₹{orig_total_cost - opt_total_cost:,.2f}", delta=f"{((orig_total_cost - opt_total_cost)/orig_total_cost)*100:.1f}%")
        col4.metric("Avg Delivery Distance", f"{df['dist_to_wh_km'].mean():.2f} km")

        st.subheader("Warehouse Capacity & Utilization")
        wh_stats = df.groupby('assigned_wh_id').agg(
            total_orders_handled=('Daily Orders', lambda x: x.astype(float).sum()),
            zones_served=('Demand Zone', 'count'),
            avg_delivery_dist=('dist_to_wh_km', 'mean')
        ).reset_index()
        
        # Add generated Lat/Lon coordinates to the utilization table
        wh_stats['Lat'] = [wh_coords[int(wid.split(" ")[1])-1][0] for wid in wh_stats['assigned_wh_id']]
        wh_stats['Lon'] = [wh_coords[int(wid.split(" ")[1])-1][1] for wid in wh_stats['assigned_wh_id']]
        
        if capacity_limit:
            wh_stats['utilization_%'] = (wh_stats['total_orders_handled'] / capacity_limit) * 100
            
        st.dataframe(wh_stats.style.format({'avg_delivery_dist': '{:.2f}', 'Lat': '{:.4f}', 'Lon': '{:.4f}', 'utilization_%': '{:.1f}'}), use_container_width=True)

    with tab3:
        st.subheader("Customer Zone Assignment Breakdown")
        display_df = df[['Demand Zone', 'Daily Orders', 'assigned_wh_id', 'dist_to_wh_km', 'delivery_cost_inr']].copy()
        display_df.rename(columns={
            'assigned_wh_id': 'Assigned Warehouse',
            'dist_to_wh_km': 'Delivery Distance (km)',
            'delivery_cost_inr': 'Delivery Cost (₹)'
        }, inplace=True)
        st.dataframe(display_df.style.format({'Delivery Distance (km)': '{:.2f}', 'Delivery Cost (₹)': '{:.2f}'}), use_container_width=True)
