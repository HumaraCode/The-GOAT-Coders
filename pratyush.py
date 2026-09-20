import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pulp
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import datetime

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer", page_icon="📍")

# --- AUTHENTICATION STATE INITIALIZATION ---
if 'user_db' not in st.session_state:
    st.session_state.user_db = {'admin': 'password123'} # Default test account
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# --- APP STATE INITIALIZATION ---
if 'locations_df' not in st.session_state:
    st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_notes' not in st.session_state:
    st.session_state.user_notes = ""

# --- THEME SWITCHER (SIDEBAR) ---
st.sidebar.header("🎨 Appearance")
app_theme = st.sidebar.selectbox("Select Theme", ["Light Mode", "Dark Mode", "Aurora Magic"])
st.sidebar.divider()

# --- DYNAMIC CSS BASED ON THEME ---
if app_theme == "Light Mode":
    theme_css = """
    <style>
        .stApp { background: #f8fafc; color: #0f172a; }
        div[data-testid="metric-container"] { background: #ffffff !important; border: 1px solid #e2e8f0; }
        .history-card { background: #f1f5f9; color: #334155; border-left: 5px solid #3b82f6; }
        h1, h2, h3 { color: #1e293b !important; }
        .streamlit-expanderHeader { background-color: #ffffff !important; color: #0f172a !important; }
        .auth-container { background: #ffffff; padding: 30px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
    </style>
    """
    map_tiles = "OpenStreetMap"
    
elif app_theme == "Dark Mode":
    theme_css = """
    <style>
        .stApp { background: #0b0f19; color: #e2e8f0; }
        div[data-testid="metric-container"] { background: #151b2b !important; border: 1px solid #1e293b; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        div[data-testid="metric-container"] label { color: #94a3b8 !important; }
        div[data-testid="metric-container"] div { color: #f8fafc !important; }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { background-color: #151b2b !important; color: #f8fafc !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
        .stTextArea textarea { background-color: #151b2b !important; color: #f8fafc !important; border: 1px solid #334155 !important; }
        .history-card { background: #151b2b; color: #cbd5e1; border-left: 5px solid #6366f1; border: 1px solid #1e293b; }
        h1 { background: -webkit-linear-gradient(45deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3, h4, label, p, span { color: #e2e8f0 !important; }
        .streamlit-expanderHeader { background-color: #151b2b !important; color: #e2e8f0 !important; border: 1px solid #1e293b !important; }
        [data-testid="stSidebar"] { background-color: #0b0f19 !important; border-right: 1px solid #1e293b !important; }
        [data-testid="stDataFrame"] { background-color: #151b2b; border-radius: 8px; }
        iframe { filter: invert(95%) hue-rotate(180deg) contrast(85%) !important; border-radius: 12px; }
        .auth-container { background: #151b2b; padding: 30px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid #1e293b; }
    </style>
    """
    map_tiles = "OpenStreetMap"
    
else:  # Aurora Magic
    theme_css = """
    <style>
        .stApp { background: linear-gradient(-45deg, #ff9a9e, #fecfef, #a1c4fd, #c2e9fb); background-size: 400% 400%; animation: gradientBG 15s ease infinite; }
        @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.4) !important; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.8); }
        .history-card { background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%); color: #1e293b; border-left: 5px solid #6366f1; }
        h1 { background: -webkit-linear-gradient(45deg, #333399, #ff00cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3, h4 { color: #333399 !important; }
        .streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.4) !important; color: #000 !important; }
        .stTextInput input, .stNumberInput input, .stTextArea textarea { background-color: rgba(255, 255, 255, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.8) !important; }
        .auth-container { background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(15px); padding: 30px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    </style>
    """
    map_tiles = "OpenStreetMap"

# Apply global CSS
st.markdown(theme_css + """
<style>
    .stButton>button {
        background-image: linear-gradient(to right, #FF416C 0%, #FF4B2B 100%) !important; color: white !important;
        border: none !important; border-radius: 50px !important; box-shadow: 0 6px 15px rgba(255, 75, 43, 0.4), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important;
        transition: all 0.15s ease !important; font-weight: 700 !important; letter-spacing: 0.5px; padding: 0.5rem 2rem !important; text-transform: uppercase;
    }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(255, 75, 43, 0.6), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important; background-image: linear-gradient(to right, #FF4B2B 0%, #FF416C 100%) !important; }
    .stButton>button:active { transform: translateY(4px) !important; box-shadow: 0 2px 5px rgba(255, 75, 43, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25) !important; }
    div[data-testid="stSidebar"] div:has(> button[kind="secondary"]) button { background-image: linear-gradient(to right, #11998e 0%, #38ef7d 100%) !important; box-shadow: 0 6px 15px rgba(56, 239, 125, 0.4), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important; }
    div[data-testid="stSidebar"] div:has(> button[kind="secondary"]) button:hover { background-image: linear-gradient(to right, #38ef7d 0%, #11998e 100%) !important; }
    div[data-testid="metric-container"] { border-radius: 16px; padding: 20px; transition: transform 0.3s ease; }
    div[data-testid="metric-container"]:hover { transform: translateY(-4px); }
    .history-card { border-radius: 12px; padding: 12px; margin-bottom: 12px; font-size: 0.9rem; }
    .stTextArea textarea { border-radius: 12px !important; font-size: 1rem !important; padding: 15px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# AUTHENTICATION SCREEN
# ==========================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=60)
        st.markdown("## GRIDPOINT Access")
        st.markdown("Please log in or create an account to use the optimizer.")
        
        tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Sign Up"])
        
        with tab_login:
            login_user = st.text_input("Username", key="log_user")
            login_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Log In", use_container_width=True):
                if login_user in st.session_state.user_db and st.session_state.user_db[login_user] == login_pass:
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab_signup:
            new_user = st.text_input("Choose Username", key="reg_user")
            new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True):
                if new_user in st.session_state.user_db:
                    st.error("Username already exists. Please choose another.")
                elif new_user == "" or new_pass == "":
                    st.warning("Please fill in both fields.")
                else:
                    st.session_state.user_db[new_user] = new_pass
                    st.success("Account created! You can now log in.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop() # Stop rendering the rest of the app if not logged in


# ==========================================
# MAIN APPLICATION (Only visible if logged in)
# ==========================================

# --- LOGOUT BUTTON IN SIDEBAR ---
st.sidebar.markdown(f"👤 **Logged in as:** `{st.session_state.current_user}`")
if st.sidebar.button("🚪 Log Out", type="secondary"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun()
st.sidebar.divider()

# --- HEADER & INTRO ---
with st.container():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=80) 
    with col2:
        st.title("GRIDPOINT Optimizer")
        st.markdown("**AI-powered geographic routing and warehouse placement.**")

st.info("💡 **Tip:** Add customer locations below, set your constraints in the sidebar, and let the algorithm find the mathematical center of gravity for your supply chain.")

# --- SIDEBAR: CONFIGURATION, HISTORY & NOTES ---
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

st.sidebar.divider()
st.sidebar.header("📝 My Strategy Notes")
current_notes = st.sidebar.text_area("Jot down ideas...", value=st.session_state.user_notes, height=150, placeholder="e.g. Look at real estate near Bangalore East...")
if st.sidebar.button("💾 Save Notes", key="save_notes_sidebar", type="secondary", use_container_width=True):
    st.session_state.user_notes = current_notes
    st.sidebar.success("Notes saved!")

st.sidebar.divider()
st.sidebar.header("🕰️ Run History")
if not st.session_state.history:
    st.sidebar.info("Your past optimization runs will appear here.")
else:
    for run in reversed(st.session_state.history[-5:]):
        st.sidebar.markdown(f"""
        <div class="history-card">
            <div style="font-weight: 800; margin-bottom: 4px; font-size: 1rem;">🕒 {run['time']}</div>
            📍 Zones Analyzed: <b>{run['zones']}</b><br>
            🏢 Warehouses Placed: <b>{run['wh_placed']}</b>
        </div>
        """, unsafe_allow_html=True)
    if st.sidebar.button("🗑️ Clear History", key="clear_hist"):
        st.session_state.history = []
        st.rerun()

# --- DATA INPUT SECTION ---
st.header("Step 1: Add Customer Demand Zones")
st.markdown("Search for regions where your orders originate. The system will plot them automatically.")

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
                st.info("Select the exact match from the results below:")
                selected_address = st.selectbox("Matching Results:", list(options.keys()))
                
                with col_orders:
                    daily_orders = st.number_input("Daily Orders", min_value=1, value=100, step=50)
                with col_btn:
                    st.write("") 
                    st.write("") 
                    if st.button("➕ Add Zone", type="primary", use_container_width=True):
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
                st.warning("No locations found. Try being less specific (e.g., just the city name).")
        except GeocoderTimedOut:
            st.error("Location service timed out. Please try again.")
        except Exception as e:
            st.error(f"Error connecting to location services: {e}")

if not st.session_state.locations_df.empty:
    st.markdown("### 📋 Current Order Demand")
    edited_df = st.data_editor(
        st.session_state.locations_df, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True,
        disabled=["Lat", "Lon", "Full Address"] 
    )
    st.session_state.locations_df = edited_df
    
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Full Address', 'Lat', 'Lon', 'Daily Orders'])
            st.session_state.opt_results = None
            st.rerun()
else:
    st.info("👋 Your network is currently empty. Use the search bar above to add customer locations.")

# --- OPTIMIZATION ALGORITHMS ---
def compute_distance_matrix(coords1, coords2):
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
    
    min_lat, max_lat = cust_coords[:, 0].min(), cust_coords[:, 0].max()
    min_lon, max_lon = cust_coords[:, 1].min(), cust_coords[:, 1].max()
    
    lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.05
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.05
    
    lat_grid = np.linspace(min_lat - lat_pad, max_lat + lat_pad, 6)
    lon_grid = np.linspace(min_lon - lon_pad, max_lon + lon_pad, 6)
    
    candidates = [[lat, lon] for lat in lat_grid for lon in lon_grid]
    
    total_orders = orders.sum()
    cg_lat = np.sum(cust_coords[:, 0] * orders) / total_orders
    cg_lon = np.sum(cust_coords[:, 1] * orders) / total_orders
    candidates.append([cg_lat, cg_lon])
    
    cand_coords = np.array(candidates)
    num_cand = len(cand_coords)
    
    dist_matrix = compute_distance_matrix(cust_coords, cand_coords)
    
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
st.header("Step 2: Generate Network Solution")

if st.button("🚀 Calculate Optimal Independent Warehouse Locations", type="primary", use_container_width=True):
    if len(st.session_state.locations_df) < 2:
        st.error("⚠️ Please add at least TWO customer demand zones above to run the optimization.")
    else:
        df_copy = st.session_state.locations_df.copy()
        
        with st.spinner("⚙️ Generating geographic grid and evaluating millions of route combinations..."):
            wh_idx, wh_coords, assignments, dist_matrix, is_optimal = solve_warehouse_placement(
                df_copy, max_warehouses, exact_k, capacity_limit, max_radius_km, cost_per_km, setup_cost
            )

        if not is_optimal:
            st.error("❌ Optimization failed. Your constraints (like capacity limits or radius) are too strict to serve all customers. Try loosening them.")
            st.session_state.opt_results = None
        else:
            st.success("✅ Optimization Complete! AI has pinpointed the ideal coordinates for your new warehouse(s).")
            st.balloons()
            
            run_record = {
                "time": datetime.datetime.now().strftime("%I:%M:%S %p"),
                "zones": len(df_copy),
                "wh_placed": len(wh_coords)
            }
            st.session_state.history.append(run_record)

            st.session_state.opt_results = {
                'df': df_copy, 'wh_idx': wh_idx, 'wh_coords': wh_coords,
                'assignments': assignments, 'dist_matrix': dist_matrix
            }

# --- RENDER RESULTS ---
if st.session_state.opt_results is not None:
    st.divider()
    st.header("📊 Step 3: Analysis & Results")
    
    res = st.session_state.opt_results
    df = res['df'].copy()
    wh_idx = res['wh_idx']
    wh_coords = res['wh_coords']
    assignments = res['assignments']
    dist_matrix = res['dist_matrix']
    
    wh_mapping = {original_j: idx for idx, original_j in enumerate(wh_idx)}
    mapped_assignments = [wh_mapping[j] for j in assignments]
    
    df['assigned_wh_id'] = [f"Warehouse {a + 1}" for a in mapped_assignments]
    df['dist_to_wh_km'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]
    df['delivery_cost_inr'] = df['dist_to_wh_km'] * df['Daily Orders'].astype(float) * cost_per_km

    opt_total_delivery = df['delivery_cost_inr'].sum()
    opt_setup_cost = len(wh_coords) * setup_cost
    opt_total_cost = opt_total_delivery + opt_setup_cost

    cust_coords = df[['Lat', 'Lon']].astype(float).values
    orders = df['Daily Orders'].astype(float).values
    cg_lat = np.sum(cust_coords[:, 0] * orders) / orders.sum()
    cg_lon = np.sum(cust_coords[:, 1] * orders) / orders.sum()
    cg_dist_matrix = compute_distance_matrix(cust_coords, np.array([[cg_lat, cg_lon]]))
    orig_delivery = sum(cg_dist_matrix[i, 0] * orders[i] * cost_per_km for i in range(len(df)))
    orig_total_cost = orig_delivery + setup_cost

    tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "💰 Financial Impact", "📋 Routing Breakdown"])

    with tab1:
        map_center = [df['Lat'].astype(float).mean(), df['Lon'].astype(float).mean()]
        m = folium.Map(location=map_center, zoom_start=11, tiles=map_tiles)
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

        if enable_radius:
            for idx, coord in enumerate(wh_coords):
                folium.Circle(
                    radius=max_radius_km * 1000,
                    location=[coord[0], coord[1]],
                    color=colors[idx % len(colors)],
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)

        for idx, coord in enumerate(wh_coords):
            color = colors[idx % len(colors)]
            folium.Marker(
                location=[coord[0], coord[1]],
                popup=f"<b>🏢 NEW WAREHOUSE {idx + 1}</b><br>Lat: {coord[0]:.4f}<br>Lon: {coord[1]:.4f}",
                icon=folium.Icon(color="black", icon_color=color, icon="building", prefix="fa")
            ).add_to(m)

        for i, row in df.iterrows():
            a_id = mapped_assignments[i]
            color = colors[a_id % len(colors)]
            wh_lat, wh_lon = wh_coords[a_id][0], wh_coords[a_id][1]
            orders_val = float(row['Daily Orders'])

            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=6 + (orders_val / 150),
                popup=f"<b>🛒 {row['Demand Zone']}</b><br>Orders: {orders_val}<br>Delivery Cost: ₹{row['delivery_cost_inr']:,.2f}",
                color=color, fill=True, fill_opacity=0.8
            ).add_to(m)

            folium.PolyLine(
                locations=[[row['Lat'], row['Lon']], [wh_lat, wh_lon]],
                color=color, weight=2, opacity=0.5, dash_array='5, 5'
            ).add_to(m)

        st_folium(m, width=1200, height=500)

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Cost", f"₹{orig_total_cost:,.2f}")
        col2.metric("Optimized Cost", f"₹{opt_total_cost:,.2f}")
        col3.metric("Total Saved", f"₹{orig_total_cost - opt_total_cost:,.2f}", delta=f"{((orig_total_cost - opt_total_cost)/orig_total_cost)*100:.1f}%")
        col4.metric("Avg Distance", f"{df['dist_to_wh_km'].mean():.2f} km")

        st.markdown("#### 🏢 Recommended Facilities Overview")
        wh_stats = df.groupby('assigned_wh_id').agg(
            total_orders_handled=('Daily Orders', lambda x: x.astype(float).sum()),
            zones_served=('Demand Zone', 'count'),
            avg_delivery_dist=('dist_to_wh_km', 'mean')
        ).reset_index()
        
        wh_stats['Suggested Lat'] = [wh_coords[int(wid.split(" ")[1])-1][0] for wid in wh_stats['assigned_wh_id']]
        wh_stats['Suggested Lon'] = [wh_coords[int(wid.split(" ")[1])-1][1] for wid in wh_stats['assigned_wh_id']]
        
        cfg = {
            "assigned_wh_id": st.column_config.TextColumn("Warehouse ID"),
            "total_orders_handled": st.column_config.NumberColumn("Total Orders", format="%d 📦"),
            "avg_delivery_dist": st.column_config.NumberColumn("Avg Distance (km)", format="%.2f km"),
            "Suggested Lat": st.column_config.NumberColumn("Suggested Lat", format="%.4f"),
            "Suggested Lon": st.column_config.NumberColumn("Suggested Lon", format="%.4f"),
            "zones_served": st.column_config.NumberColumn("Zones Served")
        }

        if capacity_limit:
            wh_stats['Capacity Utilized %'] = (wh_stats['total_orders_handled'] / capacity_limit) * 100
            cfg["Capacity Utilized %"] = st.column_config.ProgressColumn("Capacity Usage", format="%f%%", min_value=0, max_value=100)
            
        st.dataframe(wh_stats, column_config=cfg, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("#### 🚚 Customer Delivery Assignments")
        display_df = df[['Demand Zone', 'Daily Orders', 'assigned_wh_id', 'dist_to_wh_km', 'delivery_cost_inr']].copy()
        display_df.rename(columns={'assigned_wh_id': 'Assigned Warehouse', 'dist_to_wh_km': 'Delivery Distance (km)', 'delivery_cost_inr': 'Delivery Cost (₹)'}, inplace=True)
        st.dataframe(display_df.style.format({'Delivery Distance (km)': '{:.2f}', 'Delivery Cost (₹)': '{:.2f}'}), use_container_width=True)
  
