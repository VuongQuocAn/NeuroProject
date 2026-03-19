from database import SessionLocal
from models import User
from utils import hash_password

db = SessionLocal()
# Kiểm tra xem đã có admin chưa
existing_user = db.query(User).filter(User.username == "admin").first()

if not existing_user:
    new_user = User(
        username="admin",
        hashed_password=hash_password("123456"), # Mật khẩu thật sẽ là 123456
        role="researcher"
    )
    db.add(new_user)
    db.commit()
    print("Đã tạo User admin thành công! (Pass: 123456)")
else:
    print("User admin đã tồn tại trong Database.")

db.close()
