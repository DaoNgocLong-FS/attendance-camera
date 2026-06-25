"""
app.py — Web Dashboard Chấm công (Flask)
========================================
Đọc cơ sở dữ liệu attendance.db (do camera_demo.py ghi) và hiển thị dashboard
check-in / check-out một cách khoa học.

NGUYÊN LÝ (quan trọng):
    Camera ghi mỗi lần nhận diện thành MỘT dòng trong bảng `attendance`
    (timestamp, sim_score, live_score). Đây là "log sự kiện thô".

    Web app này KHÔNG hiển thị log thô. Thay vào đó nó SUY RA phiên làm việc:
        Với mỗi (nhân viên, ngày):
            - Lần nhận diện ĐẦU trong ngày  -> GIỜ VÀO  (check-in)
            - Lần nhận diện CUỐI trong ngày -> GIỜ RA   (check-out)
            - Số giờ làm = giờ ra - giờ vào
    => Cho biết mỗi người vào/ra lúc nào trong ngày, không cần sửa pipeline camera.

Chạy:
    pip install flask
    python app.py
    # hoặc:  python app.py --db D:\attendance-camera\attendance.db

Mở trình duyệt: http://localhost:5000
"""
from flask import Flask, jsonify, render_template, request
import sqlite3
import os
import sys
from datetime import datetime, date, timedelta

app = Flask(__name__)

# Cửa sổ giờ hiển thị trên trục thời gian (6h sáng -> 22h tối)
DAY_START_HOUR = 6
DAY_END_HOUR = 22

# Khoảng cách tối thiểu giữa lần đầu và lần cuối để coi là đã "tan ca".
# Nếu một người chỉ thoáng qua camera (vài giây) thì coi như mới chỉ check-in.
MIN_SESSION_SECONDS = 60


def get_db_path():
    if "--db" in sys.argv:
        i = sys.argv.index("--db")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("ATTENDANCE_DB", "attendance.db")


DB_PATH = get_db_path()


# ---------------- DB helpers (tự dò schema để linh hoạt) ----------------
def connect():
    return sqlite3.connect(DB_PATH)


def list_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows if not r[0].startswith("sqlite_")]


def columns_of(conn, t):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]


def find_col(cols, *keys):
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys):
            return c
    return None


def parse_dt(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    s = str(s)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
    return None


def fmt_duration(seconds):
    """123 phút -> '2h 03m'. None/âm -> '—'."""
    if seconds is None or seconds < 0:
        return "—"
    m = int(round(seconds / 60))
    h, m = divmod(m, 60)
    if h and m:
        return f"{h}h {m:02d}m"
    if h:
        return f"{h}h"
    return f"{m}m"


# ---------------- Suy ra phiên làm việc từ log sự kiện ----------------
def build_sessions(conn):
    """
    Đọc bảng employees + attendance (log thô), suy ra danh sách phiên:
      mỗi (employee, work_date) -> 1 phiên với check_in / check_out / duration.

    Trả về:
      sessions: list dict đã sort theo (date desc, check_in)
      employees: list dict thông tin nhân viên (đã ẩn embedding)
    """
    tables = list_tables(conn)

    # --- Tìm bảng nhân viên (có cột name + code) ---
    emp_table = None
    for t in tables:
        cols = columns_of(conn, t)
        if find_col(cols, "name") and find_col(cols, "code", "id"):
            # ưu tiên bảng có cột embedding (đúng là bảng employees)
            if find_col(cols, "embed", "blob") or t.lower().startswith("employee"):
                emp_table = t
                break
    if emp_table is None:
        for t in tables:
            if find_col(columns_of(conn, t), "name"):
                emp_table = t
                break

    # --- Tìm bảng log chấm công (có cột thời gian + employee_id) ---
    att_table = None
    for t in tables:
        if t == emp_table:
            continue
        cols = columns_of(conn, t)
        if find_col(cols, "timestamp", "time", "date") and find_col(cols, "employee", "emp_id", "code"):
            att_table = t
            break

    # Map id -> (code, name) từ bảng nhân viên
    id2emp = {}
    employees = []
    if emp_table:
        cols = columns_of(conn, emp_table)
        id_col = find_col(cols, "id") or "id"
        code_col = find_col(cols, "code") or id_col
        name_col = find_col(cols, "name") or code_col
        created_col = find_col(cols, "created", "enroll", "date")
        keep = [c for c in cols if not any(x in c.lower()
                for x in ("embed", "vector", "feature", "blob", "dim", "num_"))]
        for r in conn.execute(f'SELECT * FROM "{emp_table}"').fetchall():
            row = dict(zip(cols, r))
            eid = row.get(id_col)
            code = row.get(code_col, "")
            name = row.get(name_col, "")
            id2emp[eid] = (str(code), str(name))
            id2emp[str(code)] = (str(code), str(name))  # phòng khi log lưu code
            employees.append({
                "code": str(code),
                "name": str(name),
                "enrolled_at": (parse_dt(row.get(created_col)).strftime("%Y-%m-%d %H:%M")
                                if created_col and parse_dt(row.get(created_col)) else "—"),
                "_id": eid,
            })

    # Gom log theo (employee, ngày)
    # groups[(emp_key, date)] = list[datetime]
    groups = {}
    if att_table:
        cols = columns_of(conn, att_table)
        tcol = find_col(cols, "timestamp", "time", "date")
        ecol = find_col(cols, "employee", "emp_id") or find_col(cols, "code")
        for r in conn.execute(f'SELECT * FROM "{att_table}"').fetchall():
            row = dict(zip(cols, r))
            dt = parse_dt(row.get(tcol))
            if dt is None:
                continue
            emp_key = row.get(ecol)
            d = dt.date()
            groups.setdefault((emp_key, d), []).append(dt)

    # Dựng phiên từ mỗi nhóm
    sessions = []
    for (emp_key, d), times in groups.items():
        times.sort()
        first, last = times[0], times[-1]
        span = (last - first).total_seconds()
        code, name = id2emp.get(emp_key, id2emp.get(str(emp_key), ("?", str(emp_key))))

        has_checkout = span >= MIN_SESSION_SECONDS
        in_min = first.hour * 60 + first.minute + first.second / 60.0
        out_min = (last.hour * 60 + last.minute + last.second / 60.0) if has_checkout else None

        sessions.append({
            "code": code,
            "name": name,
            "date": d.isoformat(),
            "check_in": first.strftime("%H:%M"),
            "check_out": last.strftime("%H:%M") if has_checkout else None,
            "in_min": round(in_min, 2),
            "out_min": round(out_min, 2) if out_min is not None else None,
            "duration": fmt_duration(span) if has_checkout else "—",
            "detections": len(times),
            "open": not has_checkout,  # chỉ mới thấy 1 lần / chưa thấy tan ca
        })

    sessions.sort(key=lambda s: (s["date"], s["check_in"]), reverse=True)

    # Bổ sung thống kê cho mỗi nhân viên
    by_code_days = {}
    last_seen = {}
    for s in sessions:
        by_code_days.setdefault(s["code"], set()).add(s["date"])
        if s["code"] not in last_seen or s["date"] > last_seen[s["code"]]:
            last_seen[s["code"]] = s["date"]
    for e in employees:
        e["total_days"] = len(by_code_days.get(e["code"], set()))
        e["last_seen"] = last_seen.get(e["code"], "—")
        e.pop("_id", None)

    return sessions, employees


# ---------------- Xây dữ liệu cho frontend ----------------
def build_data():
    if not os.path.exists(DB_PATH):
        return {"error": f"Chưa tìm thấy database: {DB_PATH}"}

    conn = connect()
    try:
        if not list_tables(conn):
            return {"error": "Database rỗng (chưa có bảng nào)."}
        sessions, employees = build_sessions(conn)
    finally:
        conn.close()

    today = date.today().isoformat()
    today_sessions = [s for s in sessions if s["date"] == today]

    # --- Thẻ tổng quan ---
    checked_in_today = len({s["code"] for s in today_sessions})
    checked_out_today = len({s["code"] for s in today_sessions if not s["open"]})
    n_emp = len(employees) if employees else len({s["code"] for s in sessions})

    overview = {
        "today": today,
        "cards": [
            {"label": "Nhân viên đăng ký", "value": n_emp, "icon": "👥", "color": "#4f7cff"},
            {"label": "Đã vào làm hôm nay", "value": checked_in_today, "icon": "🟢", "color": "#22c55e"},
            {"label": "Đã tan ca hôm nay", "value": checked_out_today, "icon": "🔵", "color": "#3b82f6"},
            {"label": "Đang làm việc", "value": max(checked_in_today - checked_out_today, 0), "icon": "🏢", "color": "#f59e0b"},
        ],
    }

    # --- Biểu đồ phân bố giờ VÀO LÀM (theo tất cả phiên) ---
    in_hour = {h: 0 for h in range(DAY_START_HOUR, DAY_END_HOUR + 1)}
    for s in sessions:
        h = int(s["in_min"] // 60)
        if h in in_hour:
            in_hour[h] += 1
    chart_checkin = {
        "title": "Phân bố giờ vào làm",
        "labels": [f"{h}h" for h in range(DAY_START_HOUR, DAY_END_HOUR + 1)],
        "values": [in_hour[h] for h in range(DAY_START_HOUR, DAY_END_HOUR + 1)],
    }

    # --- Biểu đồ số phiên 14 ngày gần nhất ---
    d0 = date.today()
    days = [(d0 - timedelta(days=i)) for i in range(13, -1, -1)]
    day_count = {d.isoformat(): 0 for d in days}
    for s in sessions:
        if s["date"] in day_count:
            day_count[s["date"]] += 1
    chart_days = {
        "title": "Số phiên 14 ngày qua",
        "labels": [d.strftime("%d/%m") for d in days],
        "values": [day_count[d.isoformat()] for d in days],
    }

    # --- Danh sách ngày có dữ liệu (cho bộ chọn ngày của timeline) ---
    available_dates = sorted({s["date"] for s in sessions}, reverse=True)

    return {
        "error": None,
        "db": DB_PATH,
        "today": today,
        "day_window": {"start": DAY_START_HOUR, "end": DAY_END_HOUR},
        "overview": overview,
        "today_sessions": today_sessions,
        "sessions": sessions,
        "available_dates": available_dates,
        "employees": employees,
        "chart_checkin": chart_checkin,
        "chart_days": chart_days,
    }


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
