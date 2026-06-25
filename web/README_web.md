# Web App Cham cong (Flask)

Web application doc attendance.db, hien thi dashboard check-in/out.

## Cau truc
```
web/
  app.py                 <- Flask backend (chay file nay)
  templates/
    index.html           <- giao dien web
  requirements_web.txt
```

## Cai dat & chay

```bash
pip install flask
python app.py
```
Hoac chi dinh duong dan DB:
```bash
python app.py --db D:\attendance-camera\attendance.db
```

Mo trinh duyet: http://localhost:5000

## Demo hybrid (2 terminal)

Terminal 1 - Web app:
```bash
python app.py --db D:\attendance-camera\attendance.db
```

Terminal 2 - Camera live (OpenCV, da co):
```bash
python pipeline\camera_demo.py --detector detection_model\best.pt --recognition checkpoints\recognition\last.pt --db attendance.db --sim 0.30 --cam 0
```

-> Cham cong o cua so camera, web tu cap nhat (bat "Tu lam moi" hoac bam "Lam moi").

## Tinh nang web

- Tab Tong quan: the so lieu (da vao lam / da tan ca / dang lam viec) + 2 bieu do
- Tab Lich su: bang check-in/out (gio vao, gio ra, so gio, trang thai)
- Tab Nhan vien: danh sach dang ky
- Tu dong nhan biet bang phien check-in/out trong DB
- Khong can internet (giao dien tu chua, khong dung CDN)

## Luu y
- Web app va camera chay o 2 terminal rieng, dung chung attendance.db.
- Bat "Tu lam moi (5s)" o goc tren de tu cap nhat.
