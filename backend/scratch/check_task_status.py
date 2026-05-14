import sys
import os

# Thêm thư mục hiện tại (backend) vào sys.path
sys.path.append(os.getcwd())

from database import SessionLocal
import models

task_id = 4 

db = SessionLocal()
try:
    task = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
    if task:
        print(f"Task ID: {task.id}")
        print(f"Type: {task.task_type}")
        print(f"Status: {task.status}")
        print(f"Target ID: {task.target_id}")
        print(f"Created At: {task.created_at}")
        print(f"Error: {task.error_message}")
    else:
        print(f"No task found with ID {task_id}")
finally:
    db.close()
