import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
import pulp
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import datetime
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(layout="wide", page_title="LocaGOAT - Warehouse Optimizer", page_icon="📍")

# --- SOUND EFFECT HELPER ---
def play_success_sound():
    sound_html = """
    <script>
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        // Pleasant double-chime (A5 to E6)
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime); 
        osc.frequency.setValueAtTime(1318.51, ctx.currentTime + 0.15); 
        
        // Volume fade out
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
        
        osc.start();
        osc.stop(ctx.currentTime + 0.5);
    </script>
    """
    components.html(sound_html, width=0, height=0)

# --- AUTHENTICATION STATE INITIALIZATION ---
if 'user_db' not in st.session_state:
    st.session_state.user_db = {'admin': 'password123'}
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
app_theme = st.sidebar.selectbox("Select Theme", ["Dark Mode", "Light Mode", "Aurora Magic"])
st.sidebar.divider()

# --- DYNAMIC CSS ---
if app_theme == "Light Mode":
    theme_css = """
    <style>
        .stApp { background: #f8fafc; color: #0f172a; }
        div[data-testid="metric-container"] { background: #ffffff !important; border: 1px solid #e2e8f0; }
        .history-card { background: #f1f5f9; color: #334155; border-left: 5px solid #3b82f6; }
        h1, h2, h3 { color: #1e293b !important; }
        .streamlit-expanderHeader { background-color: #ffffff !important; color: #0f172a !important; }
        .auth-container { background: #ffffff; padding: 30px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }
        .stTextArea textarea { background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; }
        label, p, span, .stMarkdown p { color: #0f172a !important; }
        [data-testid="stSidebar"] { background-color: #f1f5f9 !important; border-right: 1px solid #e2e8f0 !important; }
        [data-testid="stDataFrame"] { background-color: #ffffff; border-radius: 8px; }
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
else:
    theme_css = """
    <style>
        .stApp { background: linear-gradient(-45deg, #ff9a9e, #fecfef, #a1c4fd, #c2e9fb); background-size: 400% 400%; animation: gradientBG 15s ease infinite; }
        @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.4) !important; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.8); }
        .history-card { background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%); color: #1e293b; border-left: 5px solid #6366f1; }
        h1 { background: -webkit-linear-gradient(45deg, #333399, #ff00cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3, h4, label, p, span { color: #333399 !important; }
        .streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.4) !important; color: #000 !important; }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea { background-color: rgba(255, 255, 255, 0.8) !important; color: #0f172a !important; border: 1px solid rgba(255, 255, 255, 0.9) !important; border-radius: 8px !important; }
        .auth-container { background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(15px); padding: 30px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.2) !important; backdrop-filter: blur(15px); border-right: 1px solid rgba(255, 255, 255, 0.5) !important; }
    </style>
    """
    map_tiles = "OpenStreetMap"

st.markdown(theme_css + """
<style>
    .stButton>button { background-image: linear-gradient(to right, #FF416C 0%, #FF4B2B 100%) !important; color: white !important; border: none !important; border-radius: 50px !important; box-shadow: 0 6px 15px rgba(255, 75, 43, 0.4), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important; transition: all 0.15s ease !important; font-weight: 700 !important; letter-spacing: 0.5px; padding: 0.5rem 2rem !important; text-transform: uppercase; }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(255, 75, 43, 0.6), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important; background-image: linear-gradient(to right, #FF4B2B 0%, #FF416C 100%) !important; }
    .stButton>button:active { transform: translateY(4px) !important; box-shadow: 0 2px 5px rgba(255, 75, 43, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25) !important; }
    div[data-testid="stSidebar"] div:has(> button[kind="secondary"]) button { background-image: linear-gradient(to right, #11998e 0%, #38ef7d 100%) !important; box-shadow: 0 6px 15px rgba(56, 239, 125, 0.4), inset 0 -5px 0 rgba(0, 0, 0, 0.25) !important; }
    div[data-testid="stSidebar"] div:has(> button[kind="secondary"]) button:hover { background-image: linear-gradient(to right, #38ef7d 0%, #11998e 100%) !important; }
    div[data-testid="metric-container"] { border-radius: 16px; padding: 20px; transition: transform 0.3s ease; }
    div[data-testid="metric-container"]:hover { transform: translateY(-4px); }
    .history-card { border-radius: 12px; padding: 12px; margin-bottom: 12px; font-size: 0.9rem; }
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
        st.markdown("## LocaGOAT AI")
        st.markdown("Enterprise logistics & warehouse optimization suite.")
        
        tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Sign Up"])
        with tab_login:
            login_user = st.text_input("Username", key="log_user")
            login_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Log In", use_container_width=True):
                if login_user in st.session_state.user_db and st.session_state.user_db[login_user] == login_pass:
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with tab_signup:
            new_user = st.text_input("Choose Username", key="reg_user")
            new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True):
                if new_user in st.session_state.user_db: st.error("Username taken.")
                elif new_user and new_pass:
                    st.session_state.user_db[new_user] = new_pass
                    st.success("Account created! You can now log in.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# MAIN APPLICATION
# ==========================================
st.sidebar.markdown(f"👤 **Logged in:** `{st.session_state.current_user}`")
if st.sidebar.button("🚪 Log Out", type="secondary"):
    st.session_state.logged_in = False
    st.rerun()
st.sidebar.divider()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("🧭 Go To")
nav_selection = st.sidebar.radio("Navigate Dashboard:", ["⚙️ Configuration", "📝 My Notes", "🕰️ Run History"])
st.sidebar.divider()

# Global variables for parameters (default states to prevent undefined errors)
optimization_mode = "Fixed K Locations"
max_warehouses = 1
exact_k = True
enable_capacity = False
capacity_limit = None
fleet_type = "Standard Diesel (Base)"
traffic_cond = "Normal"
traffic_mult = 1.0
final_cost_per_km = 15.0
base_setup = 50000.0
demand_multiplier = 1.0

# --- DYNAMIC SIDEBAR CONTENT based on Navigation ---
if nav_selection == "⚙️ Configuration":
    st.sidebar.header("⚙️ Simulation Parameters")
    with st.sidebar.expander("1. Network Strategy", expanded=True):
        optimization_mode = st.radio("Warehouse Count", ["Fixed K Locations", "Cost-Based AI Selection"])
        max_warehouses = st.slider("K Warehouses", 1, 5, 1) if "Fixed" in optimization_mode else st.slider("Max Limit", 1, 5, 3)
        exact_k = "Fixed" in optimization_mode
        enable_capacity = st.checkbox("Enable Cap Limit", value=False)
        capacity_limit = st.number_input("Max Orders/Hub", 1500) if enable_capacity else None

    with st.sidebar.expander("2. Fleet & Environment", expanded=True):
        fleet_type = st.selectbox("Fleet Vehicle Type", ["Standard Diesel (Base)", "100% Electric (EV) - High Setup, Low Running", "Hybrid Fleet"])
        traffic_cond = st.select_slider("Traffic/Congestion Pattern", options=["Light", "Normal", "Heavy"])
        traffic_mult = {"Light": 0.8, "Normal": 1.0, "Heavy": 1.5}[traffic_cond]
        base_cost_km = 15.0 
        base_setup = 50000.0
        
        if fleet_type == "100% Electric (EV) - High Setup, Low Running":
            base_cost_km *= 0.6
            base_setup *= 1.4
        elif fleet_type == "Hybrid Fleet":
            base_cost_km *= 0.85
            base_setup *= 1.15
            
        final_cost_per_km = base_cost_km * traffic_mult
        st.caption(f"Calculated Per-Km Cost: ₹{final_cost_per_km:.2f}")

    with st.sidebar.expander("3. Demand Forecasting", expanded=False):
        st.write("Model future changes in customer demand.")
        demand_multiplier = st.slider("Peak Season Demand Surge", min_value=1.0, max_value=2.0, value=1.0, step=0.1)

elif nav_selection == "📝 My Notes":
    st.sidebar.header("📝 My Strategy Notes")
    current_notes = st.sidebar.text_area("Jot down ideas, locations to research, or next steps...", value=st.session_state.user_notes, height=250)
    if st.sidebar.button("💾 Save Notes", key="save_notes_sidebar", type="secondary", use_container_width=True):
        st.session_state.user_notes = current_notes
        st.toast("Notes securely saved to session!", icon="✅")

elif nav_selection == "🕰️ Run History":
    st.sidebar.header("🕰️ Detailed Run History")
    if not st.session_state.history:
        st.sidebar.info("Your past optimization runs will appear here.")
    else:
        for run in reversed(st.session_state.history[-5:]):
            st.sidebar.markdown(f"""
            <div class="history-card">
                <div style="font-weight: 800; margin-bottom: 6px;">🕒 {run['time']}</div>
                📍 <b>Searched Zones:</b> <span style="color: #6366f1;">{run['zones_list']}</span><br>
                🛒 <b>Total Vol:</b> {run['total_orders']} orders<br>
                🏢 <b>Hubs Placed:</b> {run['wh_placed']}
            </div>
            """, unsafe_allow_html=True)
        if st.sidebar.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

# --- MAIN DASHBOARD AREA ---
with st.container():
    col1, col2 = st.columns([1, 6])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=80) 
    with col2:
        st.title("LocaGOAT Optimizer")
        st.markdown("**AI-powered geographic routing, sustainability mapping, and infrastructure scaling.**")

st.header("Step 1: Network Demand Data")
col_search, col_orders, col_btn = st.columns([2, 1, 1])

@st.cache_data(ttl=3600)
def geocode_search(query):
    geolocator = Nominatim(user_agent="locagoat_optimizer_final")
    return geolocator.geocode(query, exactly_one=False, limit=5, timeout=10)

with col_search:
    search_query = st.text_input("🔍 Search Location:", placeholder="e.g. Indiranagar, Bengaluru")

if search_query:
    with st.spinner("Geocoding..."):
        try:
            results = geocode_search(search_query)
            if results:
                options = {res.address: (res.latitude, res.longitude) for res in results}
                selected_address = st.selectbox("Select exact match:", list(options.keys()))
                with col_orders:
                    daily_orders = st.number_input("Base Daily Orders", min_value=1, value=100, step=50)
                with col_btn:
                    st.write(""); st.write("") 
                    if st.button("➕ Add Zone", type="primary", use_container_width=True):
                        short_name = selected_address.split(',')[0] 
                        lat, lon = options[selected_address]
                        new_data = pd.DataFrame([{'Demand Zone': short_name, 'Lat': lat, 'Lon': lon, 'Base Orders': daily_orders}])
                        st.session_state.locations_df = pd.concat([st.session_state.locations_df, new_data], ignore_index=True)
                        st.session_state.opt_results = None 
                        st.rerun() 
            else:
                st.warning("No locations found.")
        except Exception:
            st.error("Location service failed.")

if not st.session_state.locations_df.empty:
    st.session_state.locations_df['Forecasted Orders'] = (st.session_state.locations_df['Base Orders'] * demand_multiplier).astype(int)
    edited_df = st.data_editor(st.session_state.locations_df, num_rows="dynamic", use_container_width=True, hide_index=True, disabled=["Lat", "Lon"])
    st.session_state.locations_df = edited_df
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.locations_df = pd.DataFrame(columns=['Demand Zone', 'Lat', 'Lon', 'Base Orders'])
        st.session_state.opt_results = None
        st.rerun()

# --- OPTIMIZATION ENGINE ---
def compute_distance_matrix(coords1, coords2):
    lats1, lons1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lats2, lons2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat = lats2 - lats1[:, np.newaxis]
    dlon = lons2 - lons1[:, np.newaxis]
    a = np.sin(dlat / 2.0)**2 + np.cos(lats1[:, np.newaxis]) * np.cos(lats2) * np.sin(dlon / 2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def solve_warehouse_placement(df, max_k, exact_k, cap, transport_cost, setup_cost):
    num_cust = len(df)
    cust_coords = df[['Lat', 'Lon']].astype(float).values
    orders = df['Forecasted Orders'].astype(float).values
    min_lat, max_lat, min_lon, max_lon = cust_coords[:, 0].min(), cust_coords[:, 0].max(), cust_coords[:, 1].min(), cust_coords[:, 1].max()
    
    lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.05
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.05
    lat_grid = np.linspace(min_lat - lat_pad, max_lat + lat_pad, 6)
    lon_grid = np.linspace(min_lon - lon_pad, max_lon + lon_pad, 6)
    
    candidates = [[lat, lon] for lat in lat_grid for lon in lon_grid]
    cg_lat, cg_lon = np.sum(cust_coords[:, 0] * orders) / orders.sum(), np.sum(cust_coords[:, 1] * orders) / orders.sum()
    candidates.append([cg_lat, cg_lon])
    
    cand_coords = np.array(candidates)
    num_cand = len(cand_coords)
    dist_matrix = compute_distance_matrix(cust_coords, cand_coords)
    
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("Warehouse", range(num_cand), cat='Binary')
    x = pulp.LpVariable.dicts("Assign", (range(num_cust), range(num_cand)), cat='Binary')

    prob += pulp.lpSum(orders[i] * dist_matrix[i, j] * transport_cost * x[i][j] for i in range(num_cust) for j in range(num_cand)) + pulp.lpSum(setup_cost * y[j] for j in range(num_cand))
    prob += pulp.lpSum(y[j] for j in range(num_cand)) == max_k if exact_k else pulp.lpSum(y[j] for j in range(num_cand)) <= max_k
    if not exact_k: prob += pulp.lpSum(y[j] for j in range(num_cand)) >= 1

    for i in range(num_cust):
        prob += pulp.lpSum(x[i][j] for j in range(num_cand)) == 1
        for j in range(num_cand): prob += x[i][j] <= y[j]

    if cap:
        for j in range(num_cand): prob += pulp.lpSum(orders[i] * x[i][j] for i in range(num_cust)) <= cap * y[j]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal': return None, None, None, None, False

    selected_wh_idx = [j for j in range(num_cand) if pulp.value(y[j]) == 1]
    assignments = [next(j for j in selected_wh_idx if pulp.value(x[i][j]) == 1) for i in range(num_cust)]
    return selected_wh_idx, cand_coords[selected_wh_idx], assignments, dist_matrix, True

st.divider()
if st.button("🚀 Execute AI Network Optimization", type="primary", use_container_width=True):
    if len(st.session_state.locations_df) < 2:
        st.error("⚠️ Add at least TWO customer demand zones.")
    else:
        df_copy = st.session_state.locations_df.copy()
        with st.spinner("⚙️ Running Mixed-Integer Linear Programming..."):
            wh_idx, wh_coords, assignments, dist_matrix, is_optimal = solve_warehouse_placement(df_copy, max_warehouses, exact_k, capacity_limit, final_cost_per_km, base_setup)
        
        if not is_optimal: st.error("❌ Optimization failed. Loosen capacity limits.")
        else:
            st.balloons()
            play_success_sound()
            
            # --- DETAILED HISTORY APPEND ---
            zone_names_raw = df_copy['Demand Zone'].tolist()
            zone_names_str = ", ".join(zone_names_raw)
            if len(zone_names_str) > 55:
                zone_names_str = zone_names_str[:52] + "..."
                
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%I:%M:%S %p"),
                "zones_list": zone_names_str,
                "total_orders": df_copy['Forecasted Orders'].sum(),
                "wh_placed": len(wh_coords)
            })
            
            st.session_state.opt_results = {'df': df_copy, 'wh_idx': wh_idx, 'wh_coords': wh_coords, 'assignments': assignments, 'dist_matrix': dist_matrix}

# --- RENDER RESULTS ---
if st.session_state.opt_results is not None:
    st.header("📊 Step 2: Analysis & Dashboard")
    
    res = st.session_state.opt_results
    df, wh_coords, assignments, dist_matrix = res['df'].copy(), res['wh_coords'], res['assignments'], res['dist_matrix']
    
    mapped_assignments = [{original_j: idx for idx, original_j in enumerate(res['wh_idx'])}[j] for j in assignments]
    df['Assigned Hub'] = [f"Hub {a + 1}" for a in mapped_assignments]
    df['Dist (km)'] = [dist_matrix[i, assignments[i]] for i in range(len(df))]
    df['Total Order-km'] = df['Dist (km)'] * df['Forecasted Orders'].astype(float)
    df['Delivery Cost (₹)'] = df['Total Order-km'] * final_cost_per_km

    opt_total_delivery = df['Delivery Cost (₹)'].sum()
    opt_total_cost = opt_total_delivery + (len(wh_coords) * base_setup)

    cust_coords, orders = df[['Lat', 'Lon']].astype(float).values, df['Forecasted Orders'].astype(float).values
    cg_dist = compute_distance_matrix(cust_coords, np.array([[np.sum(cust_coords[:, 0] * orders) / orders.sum(), np.sum(cust_coords[:, 1] * orders) / orders.sum()]]))
    orig_total_km = sum(cg_dist[i, 0] * orders[i] for i in range(len(df)))
    orig_total_cost = (orig_total_km * final_cost_per_km) + base_setup

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ AI Mapping", "📈 Insights", "🌱 Sustainability", "📋 Data Export"])

    with tab1:
        sw = df[['Lat', 'Lon']].min().values.tolist()
        ne = df[['Lat', 'Lon']].max().values.tolist()
        
        m = folium.Map(tiles=map_tiles)
        m.fit_bounds([sw, ne])
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

        for idx, coord in enumerate(wh_coords):
            folium.Marker(location=[coord[0], coord[1]], popup=f"<b>🏢 HUB {idx + 1}</b>", icon=folium.Icon(color="black", icon_color=colors[idx % len(colors)], icon="star")).add_to(m)

        for i, row in df.iterrows():
            c = colors[mapped_assignments[i] % len(colors)]
            lat, lon = wh_coords[mapped_assignments[i]]
            folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=6 + (float(row['Forecasted Orders'])/100), popup=f"{row['Demand Zone']}", color=c, fill=True, fill_opacity=0.8).add_to(m)
            AntPath(locations=[[lat, lon], [row['Lat'], row['Lon']]], color=c, weight=3, opacity=0.7, dash_array=[10, 20], delay=1000).add_to(m)
        
        st_folium(m, width=1200, height=500)

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Architecture", f"₹{orig_total_cost:,.0f}")
        col2.metric("AI Optimized", f"₹{opt_total_cost:,.0f}")
        col3.metric("Savings", f"₹{orig_total_cost - opt_total_cost:,.0f}", delta=f"{((orig_total_cost - opt_total_cost)/orig_total_cost)*100:.1f}%")
        col4.metric("Avg Distance/Order", f"{df['Dist (km)'].mean():.1f} km")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, values='Forecasted Orders', names='Assigned Hub', title="Hub Load Distribution", hole=0.5, color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Facility Specifications")
            stats = df.groupby('Assigned Hub').agg(Total_Orders=('Forecasted Orders', 'sum'), Zones=('Demand Zone', 'count')).reset_index()
            st.dataframe(stats, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("### 🌱 Environmental Impact Dashboard")
        st.markdown(f"**Fleet Profile:** `{fleet_type}` | **Traffic Modifier:** `{traffic_cond}`")
        
        emissions_factor = 0.15 if "Diesel" in fleet_type else (0.05 if "Hybrid" in fleet_type else 0.0)
        orig_co2 = orig_total_km * emissions_factor
        new_co2 = df['Total Order-km'].sum() * emissions_factor
        
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("CO2 Emissions (Before)", f"{orig_co2:,.1f} kg")
        ec2.metric("CO2 Emissions (Optimized)", f"{new_co2:,.1f} kg", delta=f"-{orig_co2 - new_co2:,.1f} kg", delta_color="inverse")
        
        trees_needed = int((orig_co2 - new_co2) / 21)
        ec3.metric("Equivalent Trees Planted 🌳", f"{trees_needed} trees")
        
        if "EV" in fleet_type:
            st.success("🌍 Excellent! Your 100% Electric fleet reduces direct operational emissions to absolute zero.")

    with tab4:
        display_df = df[['Demand Zone', 'Forecasted Orders', 'Assigned Hub', 'Dist (km)', 'Delivery Cost (₹)']]
        st.dataframe(display_df.style.format({'Dist (km)': '{:.2f}', 'Delivery Cost (₹)': '{:.0f}'}), use_container_width=True)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Export Enterprise CSV", data=csv, file_name='locagoat_plan.csv', mime='text/csv', type="primary")
