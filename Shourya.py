
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.spatial.distance import cdist
import pulp
import streamlit_authenticator as stauth

st.set_page_config(layout="wide", page_title="GRIDPOINT - Warehouse Optimizer")

# --- 1. INITIALIZE USER DATABASE IN SESSION STATE ---
if "user_db" not in st.session_state:
    passwords_to_hash = ['admin123', 'analyst123']
    hashed_passwords = stauth.Hasher(passwords_to_hash).generate()

    st.session_state["user_db"] = {
        "usernames": {
            "admin": {
                "email": "admin@gridpoint.com",
                "name": "Admin User",
                "password": hashed_passwords[0]
            },
            "analyst": {
                "email": "analyst@gridpoint.com",
                "name": "Data Analyst",
                "password": hashed_passwords[1]
            }
        }
    }

# --- 2. AUTHENTICATION UI (SIGN IN / SIGN UP TABS) ---
if not st.session_state.get("authentication_status"):
    st.title("📍 GRIDPOINT Access Portal")
    
    tab_signin, tab_signup, tab_forgot = st.tabs(["🔑 Sign In", "📝 Sign Up", "🔒 Forgot Password"])

    # --- TAB 1: SIGN IN ---
    with tab_signin:
        st.subheader("Sign In to your account")
        
        authenticator = stauth.Authenticate(
            st.session_state["user_db"],
            cookie_name="gridpoint_auth_cookie",
            key="gridpoint_signature_key",
            cookie_expiry_days=30
        )
        
        try:
            authenticator.login()
        except TypeError:
            authenticator.login('Sign In', 'main')

        if st.session_state.get("authentication_status") is False:
            st.error("Invalid username or password")

    # --- TAB 2: SIGN UP ---
    with tab_signup:
        st.subheader("Create a new GRIDPOINT account")
        
        new_email = st.text_input("Email", key="signup_email")
        new_username = st.text_input("Username", key="signup_username")
        new_name = st.text_input("Full Name", key="signup_name")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

        if st.button("Sign Up", type="primary"):
            if not (new_email and new_username and new_name and new_password):
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif new_username in st.session_state["user_db"]["usernames"]:
                st.error("Username already exists. Please choose another one.")
            else:
                # Hash new password and add to user database
                hashed_pw = stauth.Hasher([new_password]).generate()[0]
                st.session_state["user_db"]["usernames"][new_username] = {
                    "email": new_email,
                    "name": new_name,
                    "password": hashed_pw
                }
                st.success("Account created successfully! Switch to the **Sign In** tab to log in.")

    # --- TAB 3: FORGOT PASSWORD ---
    with tab_forgot:
        st.subheader("Reset Password")
        forgot_username = st.text_input("Enter your username", key="forgot_user")
        new_reset_pw = st.text_input("Enter new password", type="password", key="forgot_pw")
        
        if st.button("Reset Password"):
            if forgot_username in st.session_state["user_db"]["usernames"]:
                hashed_pw = stauth.Hasher([new_reset_pw]).generate()[0]
                st.session_state["user_db"]["usernames"][forgot_username]["password"] = hashed_pw
                st.success("Password updated successfully! You can now Sign In with your new password.")
            else:
                st.error("Username not found.")

# --- 3. AUTHENTICATED APP CONTENT ---
if st.session_state.get("authentication_status"):
    
    # Re-initialize authenticator for active session context
    authenticator = stauth.Authenticate(
        st.session_state["user_db"],
        cookie_name="gridpoint_auth_cookie",
        key="gridpoint_signature_key",
        cookie_expiry_days=30
    )

    # Sidebar Logout Button
    try:
        authenticator.logout("Sign Out", "sidebar")
    except TypeError:
        authenticator.logout("Sign Out", "sidebar", key="logout_btn")

    st.sidebar.write(f"Logged in as: **{st.session_state.get('name')}**")
    st.sidebar.markdown("---")

    # ==========================================
    # --- APP MAIN CONTENT ---
    # ==========================================

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
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # --- OPTIMIZATION ALGORITHM ---
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
        st.error("Optimization failed with current constraints. Try increasing warehouse capacity or max radius limits.")
