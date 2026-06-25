# Web App Cham cong (Flask)

Web application doc attendance.db, hien thi dashboard check-in/out.

## Nguyen ly (quan trong)

Camera (`pipeline/camera_demo.py`) ghi MOI lan nhan dien thanh 1 dong trong bang
`attendance` (log su kien tho). Web app KHONG hien thi log tho ma SUY RA phien lam viec:

    Voi moi (nhan vien, ngay):
      - Lan nhan dien DAU trong ngay  -> GIO VAO  (check-in)
      - Lan nhan dien CUOI trong ngay -> GIO RA   (check-out)
      - So gio lam = gio ra - gio vao

=> Cho biet moi nguoi vao/ra LUC NAO trong ngay, dung toan bo du lieu da co,
   khong can sua pipeline camera. Neu mot nguoi chi thoang qua camera (< 60 giay)
   thi coi nhu moi chi check-in (chua ghi nhan tan ca).

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
  (phan bo gio vao lam, so phien 14 ngay qua)
- Tab Dong thoi gian: TRUC THOI GIAN truc quan — moi thanh la mot phien,
  bat dau = gio vao, ket thuc = gio ra; chon ngay de xem. Day la diem nhan
  tra loi cau hoi "ai vao/ra luc nao trong ngay".
- Tab Lich su: bang check-in/out gop theo ngay (gio vao, gio ra, so gio,
  so lan nhan dien, trang thai)
- Tab Nhan vien: danh sach dang ky + so ngay di lam + lan cuoi xuat hien
- Khong can internet (giao dien tu chua, khong dung CDN)

## Luu y
- Web app va camera chay o 2 terminal rieng, dung chung attendance.db.
- Bat "Tu lam moi (5s)" o goc tren de tu cap nhat.
