from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from utils import require_role

router = APIRouter(prefix="/system", tags=["Admin"])

# Chỉ bác sĩ và nghiên cứu viên mới được truy cập
_authorized = require_role("doctor", "researcher")


# ============================================================
# GET /system/logs — Truy xuất nhật ký truy cập hệ thống
# ============================================================

@router.get("/logs", response_model=List[schemas.AccessLogResponse])
def get_access_logs(
    limit: int = Query(default=100, ge=1, le=1000, description="Số bản ghi tối đa trả về"),
    offset: int = Query(default=0, ge=0, description="Bỏ qua N bản ghi đầu (phân trang)"),
    user_id: Optional[int] = Query(default=None, description="Lọc theo ID người dùng cụ thể"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(_authorized),
):
    """
    Truy xuất nhật ký truy cập API (Access Log).

    - Trả về danh sách log sắp xếp theo thời gian giảm dần (mới nhất trước).
    - Hỗ trợ phân trang qua `limit` và `offset`.
    - Hỗ trợ lọc theo `user_id` nếu muốn theo dõi hoạt động của một người dùng cụ thể.
    """
    query = db.query(models.AccessLog).order_by(models.AccessLog.timestamp.desc())

    if user_id is not None:
        query = query.filter(models.AccessLog.user_id == user_id)

    logs = query.offset(offset).limit(limit).all()
    return logs
