from database import SessionLocal, engine
import models
from models import User
from utils import hash_password

# Đảm bảo các bảng được tạo trước khi truy vấn
models.Base.metadata.create_all(bind=engine)

db = SessionLocal()
# Kiểm tra xem đã có admin chưa
existing_user = db.query(User).filter(User.username == "admin").first()

if not existing_user:
    new_user = User(
        username="admin",
        hashed_password=hash_password("123456"),
        role="researcher",
        is_active=True
    )
    db.add(new_user)
    db.commit()
    print("SUCCESS: Đã tạo User admin thành công! (Username: admin, Pass: 123456)")
else:
    # Reset mật khẩu nếu đã tồn tại
    existing_user.hashed_password = hash_password("123456")
    existing_user.is_active = True
    existing_user.role = "researcher"
    db.commit()
    print("SUCCESS: User admin đã tồn tại. Đã reset mật khẩu về: 123456 và đảm bảo is_active=True")

db.close()
