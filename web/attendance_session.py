"""
attendance_session.py — Logic Check-in / Check-out
===================================================
Mỗi nhân viên mỗi ngày có MỘT phiên làm việc:
  - Lần nhận diện ĐẦU trong ngày  -> CHECK-IN (vào làm), lưu giờ vào.
  - Lần nhận diện SAU (sau khoảng nghỉ cooldown) -> CHECK-OUT (tan ca), cập nhật giờ ra.

Bảng: attendance_sessions(code, name, work_date, check_in, check_out)
  UNIQUE(code, work_date) -> mỗi người 1 dòng/ngày.

Cách dùng:
    init_sessions_table(conn)
    event, t = record_attendance(conn, code, name)
    # event in {"CHECK_IN", "CHECK_OUT", "COOLDOWN"}
"""
import sqlite3
from datetime import datetime


def init_sessions_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT,
            work_date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            UNIQUE(code, work_date)
        )
    """)
    conn.commit()


def record_attendance(conn, code, name, now=None, cooldown_sec=300):
    """
    Ghi chấm công theo mô hình check-in/check-out.
    cooldown_sec: khoảng tối thiểu (giây) giữa hai lần ghi cho cùng người,
                  tránh check-out ngay sau check-in khi vẫn đứng trước camera.
                  Mặc định 300s = 5 phút (chỉnh nhỏ lại khi demo, vd 10s).

    Trả về (event, time):
      "CHECK_IN"  : vừa vào làm
      "CHECK_OUT" : vừa tan ca (đã cập nhật giờ ra = thời điểm hiện tại)
      "COOLDOWN"  : quá gần lần trước -> bỏ qua
    """
    now = now if now is not None else datetime.now()
    today = now.date().isoformat()

    row = conn.execute(
        "SELECT check_in, check_out FROM attendance_sessions WHERE code=? AND work_date=?",
        (code, today),
    ).fetchone()

    # Chưa có phiên hôm nay -> CHECK-IN
    if row is None:
        conn.execute(
            "INSERT INTO attendance_sessions(code, name, work_date, check_in) VALUES (?,?,?,?)",
            (code, name, today, now.isoformat(timespec="seconds")),
        )
        conn.commit()
        return "CHECK_IN", now

    check_in, check_out = row
    last = check_out or check_in
    last_dt = datetime.fromisoformat(last)

    # Quá gần lần trước -> bỏ qua
    if (now - last_dt).total_seconds() < cooldown_sec:
        return "COOLDOWN", None

    # Đã vào làm, giờ nhận diện lại sau cooldown -> CHECK-OUT (cập nhật giờ ra = mới nhất)
    conn.execute(
        "UPDATE attendance_sessions SET check_out=?, name=? WHERE code=? AND work_date=?",
        (now.isoformat(timespec="seconds"), name, code, today),
    )
    conn.commit()
    return "CHECK_OUT", now


def get_today_status(conn, code, now=None):
    """Trả về trạng thái hôm nay của một người: None / 'checked_in' / 'checked_out'."""
    now = now if now is not None else datetime.now()
    today = now.date().isoformat()
    row = conn.execute(
        "SELECT check_in, check_out FROM attendance_sessions WHERE code=? AND work_date=?",
        (code, today),
    ).fetchone()
    if row is None:
        return None
    check_in, check_out = row
    return "checked_out" if check_out else "checked_in"
