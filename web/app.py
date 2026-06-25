"""
app.py — Web application Chấm công (Flask)
==========================================
Web app đọc cơ sở dữ liệu attendance.db và hiển thị dashboard:
tổng quan, lịch sử check-in/out, danh sách nhân viên.

Chạy:
    pip install flask
    python app.py
    # hoặc chỉ định DB:  python app.py --db D:\attendance-camera\attendance.db

Sau đó mở trình duyệt: http://localhost:5000

Chạy SONG SONG với camera_demo.py (OpenCV) trong demo hybrid:
camera_demo.py ghi vào attendance.db, web app này đọc và hiển thị.
"""
from flask import Flask, jsonify, render_template, request
import sqlite3
import os
import sys
from datetime import datetime, date, timedelta

app = Flask(__name__)


def get_db_path():
    if "--db" in sys.argv:
        i = sys.argv.index("--db")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("ATTENDANCE_DB", "attendance.db")


DB_PATH = get_db_path()


# ---------------- DB helpers (tự dò schema) ----------------
def connect():
    return sqlite3.connect(DB_PATH)

def list_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows if not r[0].startswith("sqlite_")]

def columns_of(conn, t):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]

def find_time_col(cols):
    for c in cols:
        lc = c.lower()
        if "timestamp" in lc or "time" in lc or "date" in lc or "checkin" in lc:
            return c
    return None

def find_col(cols, *keys):
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys):
            return c
    return None

def find_session_table(conn, tables):
    for t in tables:
        lc = [c.lower() for c in columns_of(conn, t)]
        if any("check_in" in c or "checkin" in c for c in lc) and \
           any("check_out" in c or "checkout" in c for c in lc):
            return t
    return None

def find_attendance_table(conn, tables):
    for t in tables:
        if find_time_col(columns_of(conn, t)):
            return t
    return None

def find_employee_table(conn, tables, att):
    for t in tables:
        if t == att:
            continue
        lc = [c.lower() for c in columns_of(conn, t)]
        if any("name" in c for c in lc) or any(c in ("code", "id", "emp_code") for c in lc):
            return t
    return None

def rows_to_dicts(conn, table):
    cur = conn.execute(f'SELECT * FROM "{table}"')
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
    return None


# ---------------- Xây dữ liệu cho frontend ----------------
def build_data():
    if not os.path.exists(DB_PATH):
        return {"error": f"Chưa tìm thấy database: {DB_PATH}"}

    conn = connect()
    tables = list_tables(conn)
    if not tables:
        return {"error": "Database rỗng (chưa có bảng nào)."}

    session_table = find_session_table(conn, tables)
    att_table = session_table or find_attendance_table(conn, tables)
    emp_table = find_employee_table(conn, tables, att_table)

    att_rows, att_cols = rows_to_dicts(conn, att_table) if att_table else ([], [])
    emp_rows, emp_cols = rows_to_dicts(conn, emp_table) if emp_table else ([], [])

    name_col = find_col(att_cols, "name", "ten") or find_col(att_cols, "code", "id")
    today = date.today()

    data = {"mode": "session" if session_table else "log",
            "db": DB_PATH, "error": None}

    if session_table:
        ci = find_col(att_cols, "check_in", "checkin")
        co = find_col(att_cols, "check_out", "checkout")
        dcol = find_col(att_cols, "work_date", "date")

        sessions = []
        in_hours = {}      # giờ vào hôm nay
        day_counts = {}    # số phiên theo ngày
        n_in = n_out = 0
        for r in att_rows:
            din = parse_dt(r.get(ci))
            dout = parse_dt(r.get(co))
            d = (parse_dt(r.get(dcol)).date() if dcol and parse_dt(r.get(dcol)) else (din.date() if din else None))
            hours = ""
            if din and dout:
                hours = f"{(dout - din).total_seconds()/3600:.1f}h"
            sessions.append({
                "name": str(r.get(name_col, "")),
                "date": d.isoformat() if d else "",
                "check_in": din.strftime("%H:%M:%S") if din else "—",
                "check_out": dout.strftime("%H:%M:%S") if dout else "—",
                "hours": hours or "—",
                "_today": (d == today),
            })
            if d == today:
                if din:
                    n_in += 1
                    in_hours[din.hour] = in_hours.get(din.hour, 0) + 1
                if dout:
                    n_out += 1
            if d:
                day_counts[d] = day_counts.get(d, 0) + 1

        n_emp = len(emp_rows) if emp_rows else len({s["name"] for s in sessions})
        data["overview"] = {
            "cards": [
                {"label": "Nhân viên đăng ký", "value": n_emp, "icon": "👥", "color": "#4f7cff"},
                {"label": "Đã vào làm hôm nay", "value": n_in, "icon": "🟢", "color": "#22c55e"},
                {"label": "Đã tan ca hôm nay", "value": n_out, "icon": "🔵", "color": "#3b82f6"},
                {"label": "Đang làm việc", "value": max(n_in - n_out, 0), "icon": "🏢", "color": "#f59e0b"},
            ]
        }
        data["sessions"] = sorted(sessions, key=lambda s: (s["date"], s["check_in"]), reverse=True)

        # chart giờ vào hôm nay (6h-20h)
        data["chart_hour"] = {
            "title": "Giờ vào làm hôm nay",
            "labels": [f"{h}h" for h in range(6, 21)],
            "values": [in_hours.get(h, 0) for h in range(6, 21)],
        }
        # chart 7 ngày
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        data["chart_day"] = {
            "title": "Số phiên 7 ngày qua",
            "labels": [d.strftime("%d/%m") for d in days],
            "values": [day_counts.get(d, 0) for d in days],
        }

    else:
        # Chế độ log đơn giản
        tcol = find_time_col(att_cols)
        scol = find_col(att_cols, "sim", "score", "conf")
        logs = []
        hours = {}; day_counts = {}; n_today = 0; people_today = set()
        for r in att_rows:
            dt = parse_dt(r.get(tcol)) if tcol else None
            logs.append({
                "name": str(r.get(name_col, "")),
                "time": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else str(r.get(tcol, "")),
                "sim": r.get(scol, "") if scol else "",
            })
            if dt and dt.date() == today:
                n_today += 1
                people_today.add(str(r.get(name_col, "")))
                hours[dt.hour] = hours.get(dt.hour, 0) + 1
            if dt:
                day_counts[dt.date()] = day_counts.get(dt.date(), 0) + 1

        n_emp = len(emp_rows) if emp_rows else len({l["name"] for l in logs})
        data["overview"] = {"cards": [
            {"label": "Nhân viên đăng ký", "value": n_emp, "icon": "👥", "color": "#4f7cff"},
            {"label": "Lượt chấm công hôm nay", "value": n_today, "icon": "✅", "color": "#22c55e"},
            {"label": "Số người hôm nay", "value": len(people_today), "icon": "🧑", "color": "#3b82f6"},
            {"label": "Tổng lượt", "value": len(logs), "icon": "📈", "color": "#f59e0b"},
        ]}
        data["logs"] = sorted(logs, key=lambda l: l["time"], reverse=True)
        data["chart_hour"] = {"title": "Lượt chấm công hôm nay",
                              "labels": [f"{h}h" for h in range(6, 21)],
                              "values": [hours.get(h, 0) for h in range(6, 21)]}
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        data["chart_day"] = {"title": "Lượt 7 ngày qua",
                             "labels": [d.strftime("%d/%m") for d in days],
                             "values": [day_counts.get(d, 0) for d in days]}

    # Danh sách nhân viên (ẩn cột embedding)
    emp_list = []
    for r in emp_rows:
        emp_list.append({k: v for k, v in r.items()
                         if not any(x in k.lower() for x in ("embed", "vector", "feature", "blob"))})
    data["employees"] = emp_list
    conn.close()
    return data


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    return jsonify(build_data())


if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    print("Mở trình duyệt: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
