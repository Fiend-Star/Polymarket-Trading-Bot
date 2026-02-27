# import_dashboard_fixed.py
import requests
import json
from pathlib import Path
import sys
from pathlib import Path
 

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"
DASHBOARD_PATH = "dashboard.json"

def _find_existing_sa(auth):
    """Find existing service account by name. Returns ID or None."""
    resp = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/search", auth=auth)
    if resp.status_code == 200:
        for acc in resp.json().get('serviceAccounts', []):
            if acc['name'] == 'dashboard-importer':
                print(f"Found existing SA ID: {acc['id']}")
                return acc['id']
    return None

def _create_sa_token(auth, sa_id):
    """Create token for service account. Returns token string or None."""
    resp = requests.post(f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens",
                         auth=auth, json={"name": "import-token"})
    if resp.status_code == 200:
        print("✅ Token created"); return resp.json()['key']
    print(f"Error creating token: {resp.status_code}"); return None

def create_service_account_token():
    """Create a service account token (Grafana 12+)."""
    auth = (GRAFANA_USER, GRAFANA_PASS)
    resp = requests.post(f"{GRAFANA_URL}/api/serviceaccounts", auth=auth,
                         json={"name": "dashboard-importer", "role": "Admin"})
    if resp.status_code == 200:
        sa_id = resp.json()['id']
        print(f"Created SA ID: {sa_id}")
    else:
        sa_id = _find_existing_sa(auth)
        if not sa_id: return None
    return _create_sa_token(auth, sa_id)

def _load_dashboard_payload():
    """Load dashboard JSON and wrap in import payload."""
    with open(DASHBOARD_PATH, 'r') as f:
        data = json.load(f)
    if 'dashboard' in data:
        data['overwrite'] = True; return data
    return {"dashboard": data, "overwrite": True}

def import_dashboard(token):
    """Import dashboard using token."""
    print(f"Loading dashboard from {DASHBOARD_PATH}...")
    payload = _load_dashboard_payload()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=payload)
    if resp.status_code == 200:
        r = resp.json()
        print(f"✅ Dashboard imported! URL: {GRAFANA_URL}{r.get('url', '')}"); return True
    print(f"❌ Error: {resp.status_code}\n{resp.text}"); return False

def basic_auth_import():
    """Try simple basic auth import (easiest)"""
    print("Trying basic auth import...")
    
    with open(DASHBOARD_PATH, 'r') as f:
        dashboard_json = json.load(f)
    
    if 'dashboard' in dashboard_json:
        payload = dashboard_json
        payload['overwrite'] = True
    else:
        payload = {
            "dashboard": dashboard_json,
            "overwrite": True
        }
    
    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        auth=(GRAFANA_USER, GRAFANA_PASS),
        json=payload
    )
    
    if response.status_code == 200:
        print("✅ Dashboard imported with basic auth!")
        print(response.json())
        return True
    else:
        print(f"❌ Basic auth failed: {response.status_code}")
        print(response.text)
        return False

def main():
    print("=" * 60 + "\nGrafana Dashboard Importer\n" + "=" * 60)
    if not Path(DASHBOARD_PATH).exists():
        print(f"❌ Dashboard not found: {Path(DASHBOARD_PATH).absolute()}"); return False
    if basic_auth_import():
        return True
    print("\nTrying token-based auth...")
    token = create_service_account_token()
    if token and import_dashboard(token):
        return True
    print("\n❌ All methods failed. Import manually at http://localhost:3000 → + → Import")
    return False

if __name__ == "__main__":
    main()