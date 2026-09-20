"""
GRIDPOINT - Warehouse Location Optimization Platform
====================================================
Decides WHERE to open warehouses and WHICH demand zone each one serves,
minimising order-weighted delivery cost + facility setup cost.

Model: capacitated / uncapacitated facility location (MILP, solved with CBC).

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets as pysecrets
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="GRIDPOINT — Warehouse Network Optimizer",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------ optional deps
try:
    import folium
    from streamlit_folium import st_folium

    MAPS_OK = True
except Exception:  # pragma: no cover
    MAPS_OK = False

try:
    import pulp

    SOLVER_OK = True
except Exception:  # pragma: no cover
    SOLVER_OK = False

try:
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut
    from geopy.geocoders import Nominatim

    GEO_OK = True
except Exception:  # pragma: no cover
    GEO_OK = False

# ------------------------------------------------------------------ storage
DATA_DIR = Path(os.environ.get("GRIDPOINT_DATA", "gridpoint_data"))
USERS_FILE = DATA_DIR / "users.json"
STORE_DIR = DATA_DIR / "workspaces"
MAX_HISTORY = 60

EARTH_RADIUS_KM = 6371.0088
PALETTE = ["#0E7C7B", "#E0913A", "#B5446E", "#3C6E9F", "#6A8D2F", "#8E5A9B"]


def _ensure_dirs() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _write_json(path: Path, payload) -> None:
    _ensure_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(path)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)[:80]


def load_users() -> dict:
    return _read_json(USERS_FILE, {})


def save_users(users: dict) -> None:
    _write_json(USERS_FILE, users)


def workspace_path(uid: str) -> Path:
    return STORE_DIR / f"{_slug(uid)}.json"


def load_workspace(uid: str) -> dict:
    ws = _read_json(workspace_path(uid), {})
    ws.setdefault("searches", [])
    ws.setdefault("runs", [])
    ws.setdefault("plans", [])
    return ws


def save_workspace(uid: str, ws: dict) -> None:
    ws["searches"] = ws.get("searches", [])[-MAX_HISTORY:]
    ws["runs"] = ws.get("runs", [])[-MAX_HISTORY:]
    _write_json(workspace_path(uid), ws)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------- auth
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def google_configured() -> bool:
    """True only when OIDC credentials are actually present in secrets.toml."""
    if not (hasattr(st, "login") and hasattr(st, "user")):
        return False
    try:
        return "auth" in st.secrets and "google" in st.secrets["auth"]
    except Exception:
        return False


def google_identity():
    """Return the Google-authenticated identity, or None."""
    if not (hasattr(st, "user") and google_configured()):
        return None
    try:
        if not getattr(st.user, "is_logged_in", False):
            return None
        email = getattr(st.user, "email", None) or "unknown@google"
        return {
            "uid": f"google:{email}",
            "name": getattr(st.user, "name", None) or email.split("@")[0],
            "email": email,
            "provider": "Google",
        }
    except Exception:
        return None


def current_identity():
    ident = google_identity()
    if ident:
        return ident
    return st.session_state.get("identity")


def sign_out() -> None:
    ident = current_identity()
    st.session_state.pop("identity", None)
    for key in ("zones", "results", "search_hits", "plan_draft"):
        st.session_state.pop(key, None)
    if ident and ident.get("provider") == "Google":
        st.logout()
    else:
        st.rerun()


# ------------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', system-ui, sans-serif; }

.gp-mark {
  font-weight: 800; font-size: 2.6rem; color: #0F2436;
  letter-spacing: -0.03em; line-height: 1.05; margin: 0;
}
.gp-mark span { color: #0E7C7B; }
.gp-sub { color: #5B6B7B; font-size: 1.02rem; margin: .35rem 0 1.4rem 0; max-width: 62ch; }
.gp-rule { height: 3px; width: 56px; background: #E0913A; border-radius: 2px; margin: .2rem 0 1rem 0; }

.gp-card {
  border: 1px solid #E3E8EE; border-radius: 10px; padding: 1.15rem 1.3rem;
  background: #FFFFFF;
}
.gp-pill {
  display: inline-block; padding: .18rem .6rem; border-radius: 999px;
  background: #EAF4F4; color: #0E7C7B; font-size: .78rem; font-weight: 600;
}
.gp-coord { font-family: 'IBM Plex Mono', monospace; font-size: .9rem; color: #0F2436; }
.gp-note { color: #5B6B7B; font-size: .88rem; }

div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
.stButton > button { border-radius: 8px; font-weight: 600; }
section[data-testid="stSidebar"] { border-right: 1px solid #E3E8EE; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ geometry
def haversine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between every row of a and every row of b."""
    lat1 = np.radians(a[:, 0])[:, None]
    lon1 = np.radians(a[:, 1])[:, None]
    lat2 = np.radians(b[:, 0])[None, :]
    lon2 = np.radians(b[:, 1])[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def weighted_centre(coords: np.ndarray, orders: np.ndarray) -> tuple[float, float]:
    w = orders.sum()
    return float((coords[:, 0] * orders).sum() / w), float((coords[:, 1] * orders).sum() / w)


def build_candidates(coords: np.ndarray, orders: np.ndarray, grid_n: int = 9) -> np.ndarray:
    """
    Candidate warehouse sites = demand points + order-weighted centre + a grid
    over the bounding box. Demand points matter: for uncapacitated problems an
    optimal site always sits on one of them, so including them guarantees the
    solver can reach the true optimum instead of the nearest grid cell.
    """
    pts = [tuple(c) for c in coords]
    pts.append(weighted_centre(coords, orders))

    lat_lo, lat_hi = coords[:, 0].min(), coords[:, 0].max()
    lon_lo, lon_hi = coords[:, 1].min(), coords[:, 1].max()
    lat_pad = (lat_hi - lat_lo) * 0.12 or 0.05
    lon_pad = (lon_hi - lon_lo) * 0.12 or 0.05
    for la in np.linspace(lat_lo - lat_pad, lat_hi + lat_pad, grid_n):
        for lo in np.linspace(lon_lo - lon_pad, lon_hi + lon_pad, grid_n):
            pts.append((float(la), float(lo)))

    seen, unique = set(), []
    for la, lo in pts:
        key = (round(la, 4), round(lo, 4))
        if key not in seen:
            seen.add(key)
            unique.append((float(la), float(lo)))
    return np.array(unique, dtype=float)


# ----------------------------------------------------------------- optimizer
@st.cache_data(show_spinner=False, ttl=3600)
def optimise(
    zones: pd.DataFrame,
    k_max: int,
    exact_k: bool,
    capacity: float | None,
    radius_km: float | None,
    cost_per_km: float,
    setup_cost: float,
    grid_n: int,
    time_limit: int = 45,
) -> dict:
    """Solve the facility location MILP. Returns a result dict (never raises)."""
    if not SOLVER_OK:
        return {"ok": False, "reason": "PuLP is not installed — run: pip install pulp"}

    df = zones.reset_index(drop=True)
    coords = df[["Lat", "Lon"]].astype(float).to_numpy()
    orders = df["Daily Orders"].astype(float).to_numpy()
    n_zone = len(df)

    cands = build_candidates(coords, orders, grid_n=grid_n)
    dist = haversine_matrix(coords, cands)
    n_cand = len(cands)

    # --- infeasibility caught before the solver, so the message is useful ----
    if radius_km is not None:
        stranded = [
            df.loc[i, "Demand Zone"] for i in range(n_zone) if dist[i].min() > radius_km
        ]
        if stranded:
            return {
                "ok": False,
                "reason": (
                    f"No candidate site lies within {radius_km:g} km of "
                    f"{', '.join(stranded[:3])}. Raise the service radius."
                ),
            }
    if capacity is not None and orders.sum() > capacity * k_max + 1e-6:
        return {
            "ok": False,
            "reason": (
                f"Total demand is {orders.sum():,.0f} orders/day but {k_max} warehouse(s) "
                f"can only hold {capacity * k_max:,.0f}. Raise capacity or the warehouse limit."
            ),
        }

    # Assignment vars can stay continuous when there is no capacity limit: the
    # LP then naturally assigns each zone to its nearest open site, and dropping
    # n*m binaries makes the solve dramatically faster.
    x_cat = "Binary" if capacity is not None else "Continuous"

    prob = pulp.LpProblem("gridpoint_facility_location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("open", range(n_cand), cat="Binary")
    x = pulp.LpVariable.dicts(
        "serve", (range(n_zone), range(n_cand)), lowBound=0, upBound=1, cat=x_cat
    )

    prob += (
        pulp.lpSum(
            orders[i] * dist[i, j] * cost_per_km * x[i][j]
            for i in range(n_zone)
            for j in range(n_cand)
        )
        + pulp.lpSum(setup_cost * y[j] for j in range(n_cand))
    )

    if exact_k:
        prob += pulp.lpSum(y[j] for j in range(n_cand)) == k_max
    else:
        prob += pulp.lpSum(y[j] for j in range(n_cand)) <= k_max
        prob += pulp.lpSum(y[j] for j in range(n_cand)) >= 1

    for i in range(n_zone):
        prob += pulp.lpSum(x[i][j] for j in range(n_cand)) == 1
        for j in range(n_cand):
            if radius_km is not None and dist[i, j] > radius_km:
                x[i][j].upBound = 0
            else:
                prob += x[i][j] <= y[j]

    if capacity is not None:
        for j in range(n_cand):
            prob += pulp.lpSum(orders[i] * x[i][j] for i in range(n_zone)) <= capacity * y[j]

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        return {
            "ok": False,
            "reason": f"Solver returned '{status}'. Relax capacity or radius, or reduce grid density.",
        }

    open_idx = [j for j in range(n_cand) if (y[j].value() or 0) > 0.5]
    if not open_idx:
        return {"ok": False, "reason": "Solver opened no warehouse. Check your cost inputs."}

    assign = []
    for i in range(n_zone):
        best = max(open_idx, key=lambda j: (x[i][j].value() or 0))
        assign.append(best)

    wh_coords = cands[open_idx]
    slot = {j: pos for pos, j in enumerate(open_idx)}

    out = df.copy()
    out["Warehouse"] = [f"W{slot[j] + 1}" for j in assign]
    out["Distance (km)"] = [dist[i, assign[i]] for i in range(n_zone)]
    out["Delivery Cost (₹)"] = out["Distance (km)"] * orders * cost_per_km

    delivery = float(out["Delivery Cost (₹)"].sum())
    setup_total = float(setup_cost * len(open_idx))

    # Baselines for an honest comparison
    cg = np.array([weighted_centre(coords, orders)])
    cg_dist = haversine_matrix(coords, cg)[:, 0]
    baseline_cog = float((cg_dist * orders * cost_per_km).sum()) + setup_cost

    single_costs = (dist * orders[:, None] * cost_per_km).sum(axis=0)
    best_single_j = int(single_costs.argmin())
    baseline_single = float(single_costs[best_single_j]) + setup_cost

    return {
        "ok": True,
        "zones": out,
        "wh_coords": wh_coords,
        "n_wh": len(open_idx),
        "delivery_cost": delivery,
        "setup_cost": setup_total,
        "total_cost": delivery + setup_total,
        "baseline_cog": baseline_cog,
        "baseline_single": baseline_single,
        "cog": (float(cg[0][0]), float(cg[0][1])),
        "best_single": (float(cands[best_single_j][0]), float(cands[best_single_j][1])),
        "avg_km": float(np.average(out["Distance (km)"], weights=orders)),
        "max_km": float(out["Distance (km)"].max()),
        "candidates_tested": n_cand,
        "cost_per_km": cost_per_km,
        "unit_setup": setup_cost,
        "capacity": capacity,
    }


def warehouse_table(res: dict) -> pd.DataFrame:
    z = res["zones"]
    rows = []
    for idx, coord in enumerate(res["wh_coords"]):
        sub = z[z["Warehouse"] == f"W{idx + 1}"]
        rows.append(
            {
                "Warehouse": f"W{idx + 1}",
                "Latitude": round(float(coord[0]), 5),
                "Longitude": round(float(coord[1]), 5),
                "Zones served": int(len(sub)),
                "Orders/day": float(sub["Daily Orders"].astype(float).sum()),
                "Avg km": round(float(sub["Distance (km)"].mean()), 2) if len(sub) else 0.0,
                "Delivery cost (₹)": round(float(sub["Delivery Cost (₹)"].sum()), 2),
            }
        )
    out = pd.DataFrame(rows)
    if res.get("capacity"):
        out["Capacity used %"] = (out["Orders/day"] / res["capacity"] * 100).round(1)
    return out


# ------------------------------------------------------------------ geocoder
@st.cache_data(show_spinner=False, ttl=86400)
def geocode(query: str, limit: int = 6) -> list[dict]:
    if not GEO_OK:
        return []
    geo = Nominatim(user_agent="gridpoint-network-optimizer")
    hits = geo.geocode(query, exactly_one=False, limit=limit, timeout=12, addressdetails=False)
    if not hits:
        return []
    return [{"address": h.address, "lat": float(h.latitude), "lon": float(h.longitude)} for h in hits]


SAMPLE_ZONES = pd.DataFrame(
    [
        ("Indiranagar", 12.9784, 77.6408, 420),
        ("Koramangala", 12.9352, 77.6245, 610),
        ("Whitefield", 12.9698, 77.7500, 380),
        ("Jayanagar", 12.9250, 77.5938, 290),
        ("Hebbal", 13.0358, 77.5970, 210),
        ("Electronic City", 12.8452, 77.6602, 340),
        ("Rajajinagar", 12.9916, 77.5526, 180),
        ("Marathahalli", 12.9569, 77.7011, 450),
    ],
    columns=["Demand Zone", "Lat", "Lon", "Daily Orders"],
)

EMPTY_ZONES = pd.DataFrame(columns=["Demand Zone", "Lat", "Lon", "Daily Orders"])


def set_zones(df: pd.DataFrame) -> None:
    """Replace the zone table and invalidate anything derived from it."""
    st.session_state["zones"] = df.reset_index(drop=True)
    st.session_state["results"] = None
    st.session_state["zone_version"] = st.session_state.get("zone_version", 0) + 1


def init_state() -> None:
    st.session_state.setdefault("zone_version", 0)
    st.session_state.setdefault("zones", EMPTY_ZONES.copy())
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("search_hits", [])
    st.session_state.setdefault("plan_draft", "")


# =============================================================== login screen
def render_login() -> None:
    left, mid, right = st.columns([1, 1.25, 1])
    with mid:
        st.markdown('<p class="gp-mark">GRID<span>POINT</span></p>', unsafe_allow_html=True)
        st.markdown('<div class="gp-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="gp-sub">Find the coordinates where your warehouses should stand, '
            "and see which neighbourhood each one serves.</p>",
            unsafe_allow_html=True,
        )

        if google_configured():
            if st.button("Continue with Google", type="primary", use_container_width=True):
                st.login("google")
            st.markdown(
                '<p class="gp-note" style="text-align:center;margin:.6rem 0;">or use a workspace account</p>',
                unsafe_allow_html=True,
            )
        else:
            with st.expander("Turn on Google sign-in"):
                st.markdown(
                    """
Create an OAuth client at **console.cloud.google.com → Credentials → OAuth client ID
(Web application)**, add `http://localhost:8501/oauth2callback` as an authorised
redirect URI, then create `.streamlit/secrets.toml`:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "a-long-random-string"

[auth.google]
client_id = "<your-client-id>"
client_secret = "<your-client-secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Install `Authlib>=1.3.2`, restart, and the Google button appears here.
"""
                )

        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        users = load_users()

        with tab_in:
            u = st.text_input("Username", key="li_user")
            p = st.text_input("Password", type="password", key="li_pass")
            if st.button("Sign in", use_container_width=True, key="li_btn"):
                rec = users.get(u.strip().lower())
                if rec and hash_password(p, rec["salt"]) == rec["hash"]:
                    st.session_state["identity"] = {
                        "uid": f"local:{u.strip().lower()}",
                        "name": rec.get("name") or u.strip(),
                        "email": rec.get("email", ""),
                        "provider": "Workspace",
                    }
                    st.rerun()
                else:
                    st.error("That username and password don't match an account.")

        with tab_up:
            nu = st.text_input("Username", key="su_user")
            nn = st.text_input("Display name", key="su_name")
            np1 = st.text_input("Password", type="password", key="su_p1")
            np2 = st.text_input("Confirm password", type="password", key="su_p2")
            if st.button("Create account", use_container_width=True, key="su_btn"):
                key = nu.strip().lower()
                if len(key) < 3:
                    st.error("Pick a username of at least 3 characters.")
                elif key in users:
                    st.error("That username is taken. Try another.")
                elif len(np1) < 6:
                    st.error("Use a password of at least 6 characters.")
                elif np1 != np2:
                    st.error("The two passwords don't match.")
                else:
                    salt = pysecrets.token_hex(16)
                    users[key] = {
                        "salt": salt,
                        "hash": hash_password(np1, salt),
                        "name": nn.strip() or nu.strip(),
                        "created": now_stamp(),
                    }
                    save_users(users)
                    st.success("Account created. Switch to Sign in to continue.")


# ============================================================== builder page
def page_builder(uid: str, ws: dict) -> None:
    st.subheader("Demand zones")
    st.markdown(
        '<p class="gp-note">Each zone is a neighbourhood you deliver to. '
        "Order volume is the weight the optimizer pulls against.</p>",
        unsafe_allow_html=True,
    )

    search_col, action_col = st.columns([3, 1])
    with search_col:
        with st.form("zone_search", clear_on_submit=False):
            q = st.text_input("Find a neighbourhood", placeholder="Koramangala, Bengaluru")
            go = st.form_submit_button("Search", type="primary")
        if go and q.strip():
            if not GEO_OK:
                st.error("geopy isn't installed — run: pip install geopy")
            else:
                try:
                    with st.spinner("Looking up coordinates…"):
                        hits = geocode(q.strip())
                    st.session_state["search_hits"] = hits
                    ws["searches"].append(
                        {
                            "query": q.strip(),
                            "at": now_stamp(),
                            "matches": len(hits),
                            "top": hits[0]["address"] if hits else "",
                            "lat": hits[0]["lat"] if hits else None,
                            "lon": hits[0]["lon"] if hits else None,
                        }
                    )
                    save_workspace(uid, ws)
                    if not hits:
                        st.warning("Nothing matched. Add the city name and search again.")
                except (GeocoderTimedOut, GeocoderServiceError):
                    st.error("The lookup service didn't respond. Try again in a moment.")
                except Exception:
                    st.error("Couldn't reach the lookup service. Check your connection.")

    with action_col:
        st.write("")
        st.write("")
        if st.button("Load sample city", use_container_width=True):
            set_zones(SAMPLE_ZONES.copy())
            st.rerun()

    hits = st.session_state.get("search_hits") or []
    if hits:
        pick = st.selectbox("Pick the exact match", [h["address"] for h in hits])
        chosen = next(h for h in hits if h["address"] == pick)
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            label = st.text_input("Zone name", value=pick.split(",")[0].strip())
        with c2:
            vol = st.number_input("Orders per day", min_value=1, value=250, step=25)
        with c3:
            st.write("")
            st.write("")
            if st.button("Add zone", type="primary", use_container_width=True):
                row = pd.DataFrame(
                    [
                        {
                            "Demand Zone": label.strip() or pick.split(",")[0],
                            "Lat": chosen["lat"],
                            "Lon": chosen["lon"],
                            "Daily Orders": int(vol),
                        }
                    ]
                )
                set_zones(pd.concat([st.session_state["zones"], row], ignore_index=True))
                st.session_state["search_hits"] = []
                st.rerun()

    with st.expander("Add coordinates manually or from a CSV"):
        m1, m2, m3, m4 = st.columns(4)
        name_m = m1.text_input("Name", key="man_name")
        lat_m = m2.number_input("Latitude", value=12.9716, format="%.6f", key="man_lat")
        lon_m = m3.number_input("Longitude", value=77.5946, format="%.6f", key="man_lon")
        ord_m = m4.number_input("Orders/day", min_value=1, value=200, step=25, key="man_ord")
        if st.button("Add this point"):
            if not name_m.strip():
                st.error("Give the zone a name first.")
            else:
                set_zones(
                    pd.concat(
                        [
                            st.session_state["zones"],
                            pd.DataFrame(
                                [
                                    {
                                        "Demand Zone": name_m.strip(),
                                        "Lat": float(lat_m),
                                        "Lon": float(lon_m),
                                        "Daily Orders": int(ord_m),
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                )
                st.rerun()

        up = st.file_uploader("CSV with columns: Demand Zone, Lat, Lon, Daily Orders", type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                need = {"Demand Zone", "Lat", "Lon", "Daily Orders"}
                if not need.issubset(raw.columns):
                    st.error(f"The file needs these columns: {', '.join(sorted(need))}")
                else:
                    set_zones(raw[list(need)].dropna())
                    st.success(f"Loaded {len(st.session_state['zones'])} zones.")
            except Exception:
                st.error("That file couldn't be read as CSV.")

    zones = st.session_state["zones"]
    st.divider()
    if zones.empty:
        st.info("No zones yet. Search for a neighbourhood above, or load the sample city to try the model.")
        return

    edited = st.data_editor(
        zones,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Daily Orders": st.column_config.NumberColumn("Daily Orders", min_value=1, step=25),
            "Lat": st.column_config.NumberColumn("Lat", format="%.5f"),
            "Lon": st.column_config.NumberColumn("Lon", format="%.5f"),
        },
        key=f"zone_editor_{st.session_state['zone_version']}",
    )
    if not edited.equals(zones):
        st.session_state["zones"] = edited
        st.session_state["results"] = None

    clean = st.session_state["zones"].dropna(subset=["Lat", "Lon", "Daily Orders"])
    a, b, c = st.columns(3)
    a.metric("Zones", len(clean))
    b.metric("Orders per day", f"{clean['Daily Orders'].astype(float).sum():,.0f}")
    if len(clean) >= 2:
        coords = clean[["Lat", "Lon"]].astype(float).to_numpy()
        spread = haversine_matrix(coords, coords).max()
        c.metric("Network spread", f"{spread:,.1f} km")

    d1, d2 = st.columns([1, 4])
    with d1:
        if st.button("Clear all zones"):
            set_zones(EMPTY_ZONES.copy())
            st.rerun()
    with d2:
        st.download_button(
            "Download zones CSV",
            clean.to_csv(index=False).encode(),
            file_name="gridpoint_zones.csv",
            mime="text/csv",
        )


# ============================================================== results page
def render_map(res: dict) -> None:
    if not MAPS_OK:
        st.warning("Install folium and streamlit-folium to see the map: pip install folium streamlit-folium")
        return
    z = res["zones"]
    fmap = folium.Map(
        location=[z["Lat"].astype(float).mean(), z["Lon"].astype(float).mean()],
        zoom_start=11,
        tiles="CartoDB positron",
    )
    for idx, coord in enumerate(res["wh_coords"]):
        colour = PALETTE[idx % len(PALETTE)]
        folium.Marker(
            [float(coord[0]), float(coord[1])],
            tooltip=f"Warehouse W{idx + 1}",
            popup=folium.Popup(
                f"<b>Warehouse W{idx + 1}</b><br>{coord[0]:.5f}, {coord[1]:.5f}", max_width=240
            ),
            icon=folium.Icon(color="black", icon_color=colour, icon="industry", prefix="fa"),
        ).add_to(fmap)
        if res.get("radius_km"):
            folium.Circle(
                [float(coord[0]), float(coord[1])],
                radius=res["radius_km"] * 1000,
                color=colour,
                weight=1,
                fill=True,
                fill_opacity=0.04,
            ).add_to(fmap)

    for _, row in z.reset_index(drop=True).iterrows():
        wi = int(str(row["Warehouse"])[1:]) - 1
        colour = PALETTE[wi % len(PALETTE)]
        wh = res["wh_coords"][wi]
        folium.CircleMarker(
            [float(row["Lat"]), float(row["Lon"])],
            radius=5 + float(row["Daily Orders"]) ** 0.5 / 4,
            color=colour,
            fill=True,
            fill_opacity=0.85,
            weight=1,
            tooltip=f"{row['Demand Zone']} → {row['Warehouse']}",
            popup=folium.Popup(
                f"<b>{row['Demand Zone']}</b><br>{float(row['Daily Orders']):,.0f} orders/day<br>"
                f"{row['Distance (km)']:.2f} km to {row['Warehouse']}<br>"
                f"₹{row['Delivery Cost (₹)']:,.0f} per day",
                max_width=260,
            ),
        ).add_to(fmap)
        folium.PolyLine(
            [[float(row["Lat"]), float(row["Lon"])], [float(wh[0]), float(wh[1])]],
            color=colour,
            weight=2,
            opacity=0.45,
            dash_array="4,6",
        ).add_to(fmap)

    try:
        st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
    except TypeError:
        st_folium(fmap, width=1150, height=520)


def page_results(uid: str, ws: dict) -> None:
    res = st.session_state.get("results")
    if not res:
        st.info("Nothing solved yet. Build your zones, set the rules in the sidebar, then run the optimizer.")
        return

    saved = res["total_cost"]
    base = res["baseline_cog"]
    delta = base - saved

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimized daily cost", f"₹{saved:,.0f}")
    m2.metric("Single hub at centre of gravity", f"₹{base:,.0f}")
    m3.metric(
        "Saving vs that hub",
        f"₹{delta:,.0f}",
        delta=f"{(delta / base * 100):.1f}%" if base else None,
    )
    m4.metric("Weighted avg distance", f"{res['avg_km']:.2f} km")

    st.markdown(
        f'<span class="gp-pill">{res["n_wh"]} warehouse(s) opened</span> &nbsp; '
        f'<span class="gp-pill">{res["candidates_tested"]} sites evaluated</span> &nbsp; '
        f'<span class="gp-pill">longest leg {res["max_km"]:.1f} km</span>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Map", "Warehouses", "Zone assignments", "Cost breakdown"])

    with tabs[0]:
        render_map(res)

    with tabs[1]:
        wt = warehouse_table(res)
        st.dataframe(wt, use_container_width=True, hide_index=True)
        for idx, coord in enumerate(res["wh_coords"]):
            st.markdown(
                f'W{idx + 1} &nbsp; <span class="gp-coord">{coord[0]:.5f}, {coord[1]:.5f}</span> '
                f'&nbsp;·&nbsp; <a href="https://www.google.com/maps/search/?api=1&query={coord[0]:.5f},{coord[1]:.5f}" '
                f'target="_blank">open in maps</a>',
                unsafe_allow_html=True,
            )

    with tabs[2]:
        view = res["zones"][
            ["Demand Zone", "Daily Orders", "Warehouse", "Distance (km)", "Delivery Cost (₹)"]
        ].sort_values(["Warehouse", "Distance (km)"])
        st.dataframe(
            view.style.format({"Distance (km)": "{:.2f}", "Delivery Cost (₹)": "₹{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download assignments CSV",
            view.to_csv(index=False).encode(),
            file_name="gridpoint_assignments.csv",
            mime="text/csv",
        )

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Daily delivery cost by warehouse")
            wt = warehouse_table(res)
            st.bar_chart(wt.set_index("Warehouse")["Delivery cost (₹)"])
        with c2:
            st.caption("Where the money goes")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Line item": [
                            "Delivery (order-weighted)",
                            f"Setup ({res['n_wh']} × ₹{res['unit_setup']:,.0f})",
                            "Total",
                        ],
                        "Amount (₹)": [
                            round(res["delivery_cost"], 2),
                            round(res["setup_cost"], 2),
                            round(res["total_cost"], 2),
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            f"Best possible single warehouse: ₹{res['baseline_single']:,.0f} at "
            f"{res['best_single'][0]:.4f}, {res['best_single'][1]:.4f}. "
            "The centre-of-gravity hub is the usual rule-of-thumb answer; both are shown so the "
            "saving isn't measured against a straw man."
        )


# =============================================================== history page
def page_history(uid: str, ws: dict) -> None:
    runs_tab, search_tab = st.tabs(["Past optimizations", "Search history"])

    with runs_tab:
        runs = list(reversed(ws.get("runs", [])))
        if not runs:
            st.info("Your solved networks will be listed here, newest first.")
        else:
            for pos, run in enumerate(runs):
                header = (
                    f"{run['at']} · {run['n_wh']} warehouse(s) · {run['n_zones']} zones · "
                    f"₹{run['total_cost']:,.0f}/day"
                )
                with st.expander(header):
                    a, b, c = st.columns(3)
                    a.metric("Total cost", f"₹{run['total_cost']:,.0f}")
                    b.metric("Saving", f"₹{run['saving']:,.0f}")
                    c.metric("Avg distance", f"{run['avg_km']:.2f} km")
                    st.caption(
                        f"Strategy: {run['mode']} · cost/order-km ₹{run['cost_per_km']:,.2f} · "
                        f"setup ₹{run['setup_cost']:,.0f}"
                        + (f" · capacity {run['capacity']:,.0f}" if run.get("capacity") else "")
                        + (f" · radius {run['radius']:g} km" if run.get("radius") else "")
                    )
                    st.dataframe(
                        pd.DataFrame(run["warehouses"]), use_container_width=True, hide_index=True
                    )
                    if st.button("Load these zones back in", key=f"reload_{pos}"):
                        set_zones(pd.DataFrame(run["zones"]))
                        st.success("Zones restored. Head to Network to re-run them.")
            if st.button("Clear optimization history"):
                ws["runs"] = []
                save_workspace(uid, ws)
                st.rerun()

    with search_tab:
        searches = list(reversed(ws.get("searches", [])))
        if not searches:
            st.info("Every location you look up gets logged here, so you can retrace your steps.")
        else:
            sdf = pd.DataFrame(searches)[["at", "query", "top", "matches"]].rename(
                columns={"at": "When", "query": "Searched", "top": "Top match", "matches": "Results"}
            )
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            pick = st.selectbox("Re-add a past result as a zone", [s["query"] for s in searches])
            rec = next(s for s in searches if s["query"] == pick)
            if rec.get("lat") is not None:
                vol = st.number_input("Orders per day", min_value=1, value=250, step=25, key="hist_vol")
                if st.button("Add to network"):
                    set_zones(
                        pd.concat(
                            [
                                st.session_state["zones"],
                                pd.DataFrame(
                                    [
                                        {
                                            "Demand Zone": rec["top"].split(",")[0],
                                            "Lat": rec["lat"],
                                            "Lon": rec["lon"],
                                            "Daily Orders": int(vol),
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                    )
                    st.success("Added. It's waiting for you on the Network page.")
            if st.button("Clear search history"):
                ws["searches"] = []
                save_workspace(uid, ws)
                st.rerun()


# ================================================================== plan page
def plan_template(res: dict | None) -> str:
    if not res:
        return (
            "# Distribution plan\n\n"
            "## What we're solving\n\n"
            "## Proposed network\n\n"
            "## Cost impact\n\n"
            "## Rollout\n\n"
            "## Risks\n"
        )
    lines = [
        "# Distribution plan",
        f"_Drafted {now_stamp()} from a {res['n_wh']}-warehouse solution._",
        "",
        "## Proposed network",
    ]
    wt = warehouse_table(res)
    for _, r in wt.iterrows():
        lines.append(
            f"- **{r['Warehouse']}** at {r['Latitude']:.5f}, {r['Longitude']:.5f} — "
            f"{r['Zones served']} zones, {r['Orders/day']:,.0f} orders/day, avg {r['Avg km']:.2f} km."
        )
    lines += [
        "",
        "## Cost impact",
        f"- Optimized daily cost: ₹{res['total_cost']:,.0f} "
        f"(delivery ₹{res['delivery_cost']:,.0f} + setup ₹{res['setup_cost']:,.0f}).",
        f"- Centre-of-gravity hub for comparison: ₹{res['baseline_cog']:,.0f}/day.",
        f"- Saving: ₹{res['baseline_cog'] - res['total_cost']:,.0f}/day "
        f"({(res['baseline_cog'] - res['total_cost']) / res['baseline_cog'] * 100:.1f}%).",
        f"- Order-weighted average delivery distance: {res['avg_km']:.2f} km; "
        f"longest leg {res['max_km']:.2f} km.",
        "",
        "## Coverage by zone",
    ]
    for _, r in res["zones"].iterrows():
        lines.append(
            f"- {r['Demand Zone']} → {r['Warehouse']} ({r['Distance (km)']:.2f} km, "
            f"{float(r['Daily Orders']):,.0f} orders/day)"
        )
    lines += [
        "",
        "## Rollout",
        "1. ",
        "2. ",
        "",
        "## Risks and assumptions",
        "- Distances are straight-line; road factor still to be applied.",
        "- ",
    ]
    return "\n".join(lines)


def page_plan(uid: str, ws: dict) -> None:
    res = st.session_state.get("results")
    st.subheader("Write your plan")
    st.markdown(
        '<p class="gp-note">Turn the numbers into the document you actually present. '
        "Start from the solved network, then edit freely.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Start from latest results", type="primary", use_container_width=True):
            st.session_state["plan_draft"] = plan_template(res)
            st.rerun()
    with c2:
        title = st.text_input("Plan title", value=f"Distribution plan — {now_stamp()}")

    draft = st.text_area(
        "Plan (Markdown)",
        value=st.session_state.get("plan_draft", ""),
        height=420,
    )
    st.session_state["plan_draft"] = draft

    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("Save plan", use_container_width=True):
            if not draft.strip():
                st.error("Write something first, or start from the latest results.")
            else:
                ws["plans"].append({"title": title, "at": now_stamp(), "body": draft})
                save_workspace(uid, ws)
                st.success("Saved to your workspace.")
    with s2:
        st.download_button(
            "Download as Markdown",
            draft.encode(),
            file_name=f"{_slug(title) or 'plan'}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not draft.strip(),
        )
    with s3:
        preview = st.toggle("Preview", value=False)

    if preview and draft.strip():
        st.divider()
        st.markdown(draft)

    saved_plans = list(reversed(ws.get("plans", [])))
    if saved_plans:
        st.divider()
        st.markdown("#### Saved plans")
        for pos, plan in enumerate(saved_plans):
            with st.expander(f"{plan['at']} · {plan['title']}"):
                st.markdown(plan["body"])
                e1, e2 = st.columns(2)
                if e1.button("Open in editor", key=f"open_plan_{pos}"):
                    st.session_state["plan_draft"] = plan["body"]
                    st.rerun()
                if e2.button("Delete", key=f"del_plan_{pos}"):
                    ws["plans"].remove(plan)
                    save_workspace(uid, ws)
                    st.rerun()


# ======================================================================= main
def main() -> None:
    _ensure_dirs()
    init_state()

    ident = current_identity()
    if not ident:
        render_login()
        return

    uid = ident["uid"]
    ws = load_workspace(uid)

    # ---- sidebar: identity, navigation, model settings ----
    with st.sidebar:
        st.markdown(f"**{ident['name']}**")
        st.caption(f"{ident.get('email') or 'local account'} · {ident['provider']}")
        if st.button("Sign out", use_container_width=True):
            sign_out()
        st.divider()

        page = st.radio("Go to", ["Network", "Results", "History", "Plan"], key="nav")
        st.divider()

        st.markdown("**Strategy**")
        mode = st.radio(
            "How many warehouses?",
            ["Let the model decide", "Fix the number"],
            help="Fixing K forces exactly that many sites. Letting the model decide weighs each "
            "setup cost against the delivery savings it unlocks.",
        )
        if mode == "Fix the number":
            k = st.slider("Warehouses to open", 1, 6, 2)
            exact = True
        else:
            k = st.slider("Maximum warehouses", 1, 6, 4)
            exact = False

        st.markdown("**Costs**")
        cost_km = st.number_input("Transport cost per order-km (₹)", value=15.0, min_value=0.1, step=1.0)
        setup = st.number_input("Setup cost per warehouse (₹)", value=50000.0, min_value=0.0, step=5000.0)

        st.markdown("**Constraints**")
        cap_on = st.checkbox("Cap orders per warehouse")
        cap = st.number_input("Max orders per day", value=1500, min_value=1, step=100) if cap_on else None
        rad_on = st.checkbox("Cap delivery radius")
        rad = st.slider("Max service radius (km)", 2, 120, 25) if rad_on else None

        with st.expander("Solver detail"):
            grid_n = st.slider(
                "Candidate grid density",
                5,
                14,
                9,
                help="Denser grids search more locations and take longer. Your demand points are "
                "always candidates, so 9 is plenty for most networks.",
            )

    # ---- header ----
    st.markdown('<p class="gp-mark">GRID<span>POINT</span></p>', unsafe_allow_html=True)
    st.markdown('<div class="gp-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="gp-sub">Warehouse siting and zone assignment that minimise order-weighted '
        "delivery cost.</p>",
        unsafe_allow_html=True,
    )

    if not SOLVER_OK:
        st.error("PuLP isn't installed, so nothing can be solved yet. Run: pip install pulp")

    if page == "Network":
        page_builder(uid, ws)
        st.divider()
        clean = st.session_state["zones"].dropna(subset=["Lat", "Lon", "Daily Orders"])
        run = st.button(
            "Find optimal warehouse locations",
            type="primary",
            use_container_width=True,
            disabled=len(clean) < 2,
        )
        if len(clean) < 2:
            st.caption("Add at least two zones to run the model.")
        if run:
            with st.spinner("Searching candidate sites and solving…"):
                res = optimise(
                    clean.reset_index(drop=True),
                    int(k),
                    exact,
                    float(cap) if cap else None,
                    float(rad) if rad else None,
                    float(cost_km),
                    float(setup),
                    int(grid_n),
                )
            if not res.get("ok"):
                st.error(res.get("reason", "The model couldn't find a feasible network."))
                st.session_state["results"] = None
            else:
                res["radius_km"] = float(rad) if rad else None
                st.session_state["results"] = res
                ws["runs"].append(
                    {
                        "at": now_stamp(),
                        "mode": mode,
                        "n_wh": res["n_wh"],
                        "n_zones": int(len(clean)),
                        "total_cost": res["total_cost"],
                        "saving": res["baseline_cog"] - res["total_cost"],
                        "avg_km": res["avg_km"],
                        "cost_per_km": float(cost_km),
                        "setup_cost": float(setup),
                        "capacity": float(cap) if cap else None,
                        "radius": float(rad) if rad else None,
                        "warehouses": warehouse_table(res).to_dict("records"),
                        "zones": clean.to_dict("records"),
                    }
                )
                save_workspace(uid, ws)
                st.success(
                    f"Solved. {res['n_wh']} warehouse(s) across {len(clean)} zones — "
                    f"open Results for the map and the numbers."
                )
                page_results(uid, ws)

    elif page == "Results":
        page_results(uid, ws)
    elif page == "History":
        page_history(uid, ws)
    else:
        page_plan(uid, ws)


if __name__ == "__main__":
    main()
