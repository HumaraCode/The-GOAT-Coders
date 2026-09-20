import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import hashlib

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer", page_icon="📍")

st.markdown("""
<style>
    .main-title { font-size: 3.5rem; font-weight: 900; color: #1E3A8A; text-align: center; margin-bottom: 0; letter-spacing: -1px; }
    .sub-title { font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2.5rem; font-weight: 500; }
    .auth-container { max-width: 450px; margin: 0 auto; padding: 2.5rem; border-radius: 12px; background: #ffffff; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; }
    .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.6rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

if 'users' not in st.session_state:
    st.session_state.users = {'admin': hashlib.sha256('admin123'.encode()).hexdigest()}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = ""
if 'locations_df' not in st.session_state:
    st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if not st.session_state.logged_in:
    st.markdown('<p class="main-title">📍 GRIDPOINT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Intelligent Distribution Network Optimization</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔒 Sign In", "📝 Create Account"])
        
        with tab_login:
            st.markdown("#### Welcome Back")
            login_username = st.text_input("Username", key="log_user")
            login_password = st.text_input("Password", type="password", key="log_pass")
            if st.button("Sign In", type="primary", use_container_width=True):
                if login_username in st.session_state.users and st.session_state.users[login_username] == hash_password(login_password):
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_username
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.info("Demo Account: **admin** | Password: **admin123**")
            
        with tab_register:
            st.markdown("#### Join Workspace")
            reg_username = st.text_input("Choose Username", key="reg_user")
            reg_password = st.text_input("Create Password", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Register", type="primary", use_container_width=True):
                if reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_username) < 3 or len(reg_password) < 5:
                    st.error("Username must be 3+ chars and password 5+ chars.")
                elif reg_username in st.session_state.users:
                    st.error("Username already exists.")
                else:
                    st.session_state.users[reg_username] = hash_password(reg_password)
                    st.success("Account created! Please switch to Sign In.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

col_nav1, col_nav2 = st.columns([5, 1])
with col_nav1:
    st.title("📍 GRIDPOINT Workspace")
with col_nav2:
    st.write(f"👤 **{st.session_state.current_user}**")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.session_state.opt_results = None
        st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
        st.rerun()

with st.expander("❓ **Quick Start Guide & FAQ**"):
    st.markdown("""
    **What does this tool do?**  
    This AI analyzes where your customers are located (Demand Zones) and calculates the exact geographic coordinates where you should open warehouses to make deliveries as cheap and fast as possible.
    
    **How do I use it?**  
    1. Adjust your business constraints in the left sidebar (costs, limits, etc.).
    2. Add your customer locations in the main area below by searching for their city or neighborhood.
    3. Click "Calculate Optimal Warehouse Locations" to let the AI do the heavy lifting.
    
    **What is the difference between 'Fixed' and 'Cost-Based Auto-Selection'?**  
    * **Fixed:** Forces the AI to place exactly *K* warehouses, regardless of the setup cost.
    * **Cost-Based:** You set a maximum limit, and the AI decides exactly how many warehouses to open by balancing the *Setup Cost* against the *Delivery Savings*.
    """)

st.sidebar.header("⚙️ Configuration Panel")

with st.sidebar.expander("1. Strategy & Network Size", expanded=True):
    optimization_mode = st.radio("Optimization Strategy", ["Fixed Number of Warehouses", "Cost-Based Auto-Selection"])
    if optimization_mode == "Fixed Number of Warehouses":
        max_warehouses = st.slider("Warehouses to Open (K)", min_value=1, max_value=5, value=1)
        exact_k = True
    else:
        max_warehouses = st.slider("Max Allowable Warehouses", min_value=1, max_value=5, value=3)
        exact_k = False

with st.sidebar.expander("2. Financial Parameters", expanded=True):
    cost_per_km = st.number_input("Transport Cost per Order-km (₹)", value=15.0, step=1.0)
    setup_cost = st.number_input("Warehouse Setup Cost (₹)", value=50000.0, step=5000.0)

with st.sidebar.expander("3. Operational Constraints", expanded=False):
    enable_capacity = st.checkbox("Enable Capacity Limit", value=False)
    capacity_limit = st.number_input("Max Orders per Warehouse", value=1500, step=100) if enable_capacity else None
    enable_radius = st.checkbox("Enable Max Delivery Radius", value=False)
    max_radius_km = st.slider("Max Service Radius (km)", 5, 100, 20) if enable_radius else None

st.header("Step 1: Add Customer Demand Zones")
col_search, col_orders, col_btn = st.columns([2, 1, 1])

with col_search:
    search_query = st.text_input("🔍 Search Location Name:", placeholder="e.g. Indiranagar, Bengaluru")

if search_query:
    geolocator = Nominatim(user_agent="gridpoint_optimizer_final")
    with st.spinner("Finding location coordinates..."):
        try:
            results = geolocator.geocode(search_query, exactly_one=False, limit=5, timeout=10)
            if results:
                options = {res.address: (res.latitude, res.longitude) for res in results}
                selected_address = st.selectbox("Select exact match:", list(options.keys()))
                
                with col_orders:
                    daily_orders = st.number_input("Daily Orders", min_value=1, value=100, step=50)
                with col_btn:
                    st.write("") 
                    st.write("") 
                    if st.button("➕ Add Zone", type="primary", use_container_width=True):
                        short_name = selected_address.split(',')[0] 
                        lat, lon = options[selected_address]
                        new_data = pd.DataFrame([{'Demand Zone': short_name, 'Full Address': selected_address, 'Lat': lat, 'Lon': lon, 'Daily Orders': daily_orders}])
                        st.session_state.locations_df = pd.concat([st.session_state.locations_df, new_data], ignore_index=True)
                        st.session_state.opt_results = None 
                        st.rerun() 
            else:
                st.warning("No locations found.")
        except GeocoderTimedOut:
            st.error("Location service timed out.")
        except Exception:
            st.error("Error connecting to location services.")

if not st.session_state.locations_df.empty:
    st.markdown("### 📋 Current Order Demand")
    st.session_state.locations_df = st.data_editor(st.session_state.locations_df, num_rows="dynamic", use_container_width=True, hide_index=True, disabled=["Lat", "Lon", "Full Address"])
    if st.button("🗑️ Clear All Data"):
        st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
        st.session_state.opt_results = None
        st.rerun()
else:
    st.info("👋 Network is empty. Use the search bar above to map demand.")

def compute_distance_matrix(coords1, coords2):
    lats1, lons1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lats2, lons2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat, dlon = lats2 - lats1[:, np.newaxis], lons2 - lons1[:, np.newaxis]
    a = np.sin(dlat / 2.0)**2 + np.cos(lats1[:, np.newaxis]) * np.cos(lats2) * np.sin(dlon / 2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def solve_warehouse_placement(df, max_k, exact_k, cap, max_rad, transport_cost, setup_cost):
    num_cust = len(df)
    cust_coords = df[['Lat', 'Lon']].astype(float).values
    orders = df['Daily Orders'].astype(float).values
    
    min_lat, max_lat = cust_coords[:, 0].min(), cust_coords[:, 0].max()
    min_lon, max_lon = cust_coords[:, 1].min(), cust_coords[:, 1].max()
    lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.05
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.05
    
    lat_grid = np.linspace(min_lat - lat_pad, max_lat + lat_pad, 6)
    lon_grid = np.linspace(min_lon - lon_pad, max_lon + lon_pad, 6)
    candidates = [[lat, lon] for lat in lat_grid for lon in lon_grid]
    
    total_orders = orders.sum()
    candidates.append([np.sum(cust_coords[:, 0] * orders) / total_orders, np.sum(cust_coords[:, 1] * orders) / total_orders])
    cand_coords = np.array(candidates)
    num_cand = len(cand_coords)
    dist_matrix = compute_distance_matrix(cust_coords, cand_coords)
    
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("Warehouse", range(num_cand), cat='Binary')
    x = pulp.LpVariable.dicts("Assign", (range(num_cust), range(num_cand)), cat='Binary')

    prob += pulp.lpSum(orders[i] * dist_matrix[i, j] * transport_cost * x[i][j] for i in range(num_cust) for j in range(num_cand)) + pulp.lpSum(setup_cost * y[j] for j in range(num_cand))

    if exact_k: prob += pulp.lpSum(y[j] for j in range(num_cand)) == max_k
    else: prob += pulp.lpSum(y[j] for j in range(num_cand)) <= max_k; prob += pulp.lpSum(y[j] for j in range(num_cand)) >= 1

    for i in range(num_cust): prob += pulp.lpSum(x[i][j] for j in range(num_cand)) == 1
    for i in range(num_cust):
        for j in range(num_cand): prob += x[i][j] <= y[j]

    if cap is not None:
        for j in range(num_cand): prob += pulp.lpSum(orders[i] * x[i][j] for i in range(num_cust)) <= cap * y[j]
    if max_rad is not None:
        for i in range(num_cust):
            for j in range(num_cand):
                if dist_matrix[i, j] > max_rad: prob += x[i][j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal': return None, None, None, None, False

    wh_idx = [j for j in range(num_cand) if pulp.value(y[j]) == 1]
    return wh_idx, cand_coords[wh_idx], [next(j for j in wh_idx if pulp.value(x[i][j]) == 1) for i in range(num_cust)], dist_matrix, True

st.divider()
st.header("Step 2: Generate Network Solution")

if st.button("🚀 Calculate Optimal Locations", type="primary", use_container_width=True):
    if len(st.session_state.locations_df) < 2:
        st.error("⚠️ Add at least TWO customer demand zones to optimize.")
    else:
        df_copy = st.session_state.locations_df.copy()
        with st.spinner("⚙️ Evaluating route combinations..."):
            wh_idx, wh_coords, assignments, dist_matrix, is_optimal = solve_warehouse_placement(df_copy, max_warehouses, exact_k, capacity_limit, max_radius_km, cost_per_km, setup_cost)
        if not is_optimal:
            st.error("❌ Optimization failed. Loosen capacity or radius limits.")
            st.session_state.opt_results = None
        else:
            st.success("✅ AI has pinpointed ideal coordinates.")
            st.session_state.opt_results = {'df': df_copy, 'wh_idx': wh_idx, 'wh_coords': wh_coords, 'assignments': assignments, 'dist_matrix': dist_matrix}

if st.session_state.opt_results is not None:
    st.divider()
    st.header("📊 Step 3: Analysis & Results")
    
    res = st.session_state.opt_results
    df, wh_coords, assignments, dist_matrix = res['df'].copy(), res['wh_coords'], res['assignments'], res['dist_matrix']
    wh_mapping = {original_j: idx for idx, original_j in enumerate(res['wh_idx'])}
    mapped_assignments = [wh_mapping[j] for j in assignments]
    
    df['assigned_wh_id'] = [f"Warehouse {a + 1}" for a in mapped_assignments]
    df['dist_to_wh_km'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]
    df['delivery_cost_inr'] = df['dist_to_wh_km'] * df['Daily Orders'].astype(float) * cost_per_km

    opt_total_cost = df['delivery_cost_inr'].sum() + (len(wh_coords) * setup_cost)
    cust_coords, orders = df[['Lat', 'Lon']].astype(float).values, df['Daily Orders'].astype(float).values
    cg_lat, cg_lon = np.sum(cust_coords[:, 0] * orders) / orders.sum(), np.sum(cust_coords[:, 1] * orders) / orders.sum()
    orig_total_cost = sum(compute_distance_matrix(cust_coords, np.array([[cg_lat, cg_lon]]))[i, 0] * orders[i] * cost_per_km for i in range(len(df))) + setup_cost

    tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "💰 Financial Impact", "📋 Routing Breakdown"])

    with tab1:
        m = folium.Map(location=[df['Lat'].astype(float).mean(), df['Lon'].astype(float).mean()], zoom_start=11, tiles="CartoDB positron")
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231']

        for idx, coord in enumerate(wh_coords):
            folium.Marker([coord[0], coord[1]], popup=f"<b>🏢 NEW WAREHOUSE {idx + 1}</b><br>Lat: {coord[0]:.4f}<br>Lon: {coord[1]:.4f}", icon=folium.Icon(color="black", icon_color=colors[idx % len(colors)], icon="building", prefix="fa")).add_to(m)

        for i, row in df.iterrows():
            a_id = mapped_assignments[i]
            folium.CircleMarker([row['Lat'], row['Lon']], radius=6 + (float(row['Daily Orders']) / 150), popup=f"<b>🛒 {row['Demand Zone']}</b><br>Cost: ₹{row['delivery_cost_inr']:,.2f}", color=colors[a_id % len(colors)], fill=True, fill_opacity=0.8).add_to(m)
            folium.PolyLine([[row['Lat'], row['Lon']], [wh_coords[a_id][0], wh_coords[a_id][1]]], color=colors[a_id % len(colors)], weight=2, opacity=0.5, dash_array='5, 5').add_to(m)

        st_folium(m, width=1200, height=500)

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Cost (Center of Gravity)", f"₹{orig_total_cost:,.2f}")
        col2.metric("Optimized Total Cost", f"₹{opt_total_cost:,.2f}")
        col3.metric("Total Money Saved", f"₹{orig_total_cost - opt_total_cost:,.2f}", delta=f"{((orig_total_cost - opt_total_cost)/orig_total_cost)*100:.1f}%")
        col4.metric("Avg Delivery Distance", f"{df['dist_to_wh_km'].mean():.2f} km")

        wh_stats = df.groupby('assigned_wh_id').agg(total_orders_handled=('Daily Orders', lambda x: x.astype(float).sum()), zones_served=('Demand Zone', 'count'), avg_delivery_dist=('dist_to_wh_km', 'mean')).reset_index()
        wh_stats['Suggested Lat'] = [wh_coords[int(wid.split(" ")[1])-1][0] for wid in wh_stats['assigned_wh_id']]
        wh_stats['Suggested Lon'] = [wh_coords[int(wid.split(" ")[1])-1][1] for wid in wh_stats['assigned_wh_id']]
        if capacity_limit: wh_stats['Capacity Utilized %'] = (wh_stats['total_orders_handled'] / capacity_limit) * 100
        st.dataframe(wh_stats.style.format({'avg_delivery_dist': '{:.2f}', 'Suggested Lat': '{:.4f}', 'Suggested Lon': '{:.4f}', 'Capacity Utilized %': '{:.1f}'}), use_container_width=True)

    with tab3:
        display_df = df[['Demand Zone', 'Daily Orders', 'assigned_wh_id', 'dist_to_wh_km', 'delivery_cost_inr']].rename(columns={'assigned_wh_id': 'Assigned Warehouse', 'dist_to_wh_km': 'Distance (km)', 'delivery_cost_inr': 'Cost (₹)'})
        st.dataframe(display_df.style.format({'Distance (km)': '{:.2f}', 'Cost (₹)': '{:.2f}'}), use_container_width=True)
