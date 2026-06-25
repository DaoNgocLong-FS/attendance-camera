r"""
confirm_tracker.py — Bộ đếm xác nhận nhận diện liên tục
========================================================
Một người phải được nhận diện LIÊN TỤC đủ N giây (mặc định 2s) mới được tính
là chấm công. Nếu khuôn mặt biến mất giữa chừng -> reset bộ đếm.

Mục đích: tránh ghi trùng hàng loạt khi một người đứng lâu trước camera
(trước đây mỗi frame ghi 1 dòng -> hàng trăm dòng/phút). Với xác nhận liên tục,
mỗi lần xuất hiện chỉ ghi ĐÚNG MỘT lần khi vừa đủ số giây yêu cầu.

Dùng trong AttendancePipeline.process_frame:
    progress, just_confirmed = tracker.update(emp_code)
    # progress: 0..1 (vẽ thanh tiến trình)
    # just_confirmed: True đúng 1 lần khi vừa đủ N giây -> ghi chấm công
    ...
    tracker.cleanup()   # cuối mỗi frame: xoá timer của mặt đã rời khung
"""
import time


class ContinuousConfirmTracker:
    def __init__(self, required_seconds=2.0, reset_gap=1.5, forget_after=3.0):
        """
        required_seconds: số giây liên tục cần để xác nhận.
        reset_gap: không thấy mặt quá khoảng này (giây) -> coi như gián đoạn, reset bộ đếm.
        forget_after: xoá hẳn timer nếu mặt rời đi quá khoảng này (giây) -> lần sau tính phiên mới.
        """
        self.required = required_seconds
        self.reset_gap = reset_gap
        self.forget_after = forget_after
        self.timers = {}  # code -> {first, last, confirmed}

    def update(self, code, now=None):
        """
        Gọi mỗi frame cho mỗi mã đã vượt ngưỡng nhận diện.
        Trả về (progress 0..1, just_confirmed bool).
        just_confirmed = True ĐÚNG MỘT LẦN khi vừa đủ số giây yêu cầu.
        """
        now = now if now is not None else time.time()
        t = self.timers.get(code)

        # Bắt đầu chu kỳ mới nếu chưa có, hoặc bị gián đoạn quá lâu
        if t is None or (now - t["last"]) > self.reset_gap:
            self.timers[code] = {"first": now, "last": now, "confirmed": False}
            return 0.0, False

        t["last"] = now
        elapsed = now - t["first"]
        progress = min(elapsed / self.required, 1.0)

        just_confirmed = False
        if elapsed >= self.required and not t["confirmed"]:
            t["confirmed"] = True
            just_confirmed = True
        return progress, just_confirmed

    def is_confirmed(self, code):
        t = self.timers.get(code)
        return bool(t and t["confirmed"])

    def cleanup(self, now=None):
        """Xoá timer của khuôn mặt đã rời khỏi khung (để lần sau quay lại được tính phiên mới)."""
        now = now if now is not None else time.time()
        for code in list(self.timers):
            if (now - self.timers[code]["last"]) > self.forget_after:
                del self.timers[code]
