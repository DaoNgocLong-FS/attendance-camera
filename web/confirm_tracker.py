"""
confirm_tracker.py — Bộ đếm xác nhận nhận diện liên tục
========================================================
Một nhân viên phải được nhận diện LIÊN TỤC đủ N giây (mặc định 3s)
mới được tính là chấm công. Nếu khuôn mặt biến mất giữa chừng -> reset.

Cách dùng trong vòng lặp camera:
    tracker = ContinuousConfirmTracker(required_seconds=3.0)
    # mỗi frame, với mỗi mã nhân viên nhận diện được:
    progress, confirmed = tracker.update(code)
    # progress: 0..1 (để vẽ thanh tiến trình)
    # confirmed: True đúng 1 lần khi vừa đủ 3 giây -> gọi ghi chấm công
    # cuối mỗi frame:
    tracker.cleanup()  # xoá timer của mặt đã rời đi
"""
import time


class ContinuousConfirmTracker:
    def __init__(self, required_seconds=3.0, reset_gap=1.0, forget_after=2.0):
        """
        required_seconds: số giây liên tục cần để xác nhận.
        reset_gap: nếu không thấy mặt quá khoảng này (giây) -> coi như gián đoạn, reset bộ đếm.
        forget_after: xoá hẳn timer nếu mặt rời đi quá khoảng này (giây).
        """
        self.required = required_seconds
        self.reset_gap = reset_gap
        self.forget_after = forget_after
        self.timers = {}  # code -> {first, last, confirmed}

    def update(self, code, now=None):
        """
        Gọi mỗi frame cho mỗi mã nhận diện được (đã vượt ngưỡng cosine).
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
        """Xoá timer của khuôn mặt đã rời khỏi khung hình (để lần sau quay lại được tính phiên mới)."""
        now = now if now is not None else time.time()
        for code in list(self.timers):
            if (now - self.timers[code]["last"]) > self.forget_after:
                del self.timers[code]
