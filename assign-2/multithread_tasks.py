from threading_module import myThread
import asyncio
import time

"""
Runs a set of tasks in parallel using threads
"""

def main():
    task_wait_times = [0, 2, 6, 1, 3]
    threads = []

    # Start each task
    for i, task_wait_time in enumerate(task_wait_times):
        task_id = i+1
        task_name = "task%d" % (task_id)
        thread = myThread(task_id, task_name, task_wait_time)
        thread.start()
        threads.append(thread)

    # Wait for all the tasks to finish
    for thread in threads:
        thread.join()

    print ("All tasks completed")

if __name__ == '__main__':
    main()