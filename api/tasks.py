TASKS = {}

def log_task_msg(task_id: str, msg: str):
    if task_id in TASKS:
        TASKS[task_id]["messages"].append(msg)
