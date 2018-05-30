from threading_module import myThread
import asyncio
import time
import sys

"""
Runs a defined set of tasks asynchronously using threads
# Task1 needs to be run independently
# Task2 needs to be run after task1 finishes
# Task3 needs to be run after task2 finishes
# Taks4 needs to be run after task1 and task3 finish
# Task5 needs to be run independently
"""

# tasks
#   Each entry is a tuple containing:
#       ( task_name, task_id, wait_time, dependent_task_ids)
tasks = [ ('task1', 1, 0, []),
          ('task2', 2, 2, [1]),
          ('task3', 3, 6, [2]),
          ('task4', 4, 1, [1, 3]),
          ('task5', 5, 3, [])
        ]

threads = []

def main():
    # Start each task according to defined dependencies
    for task in tasks:
        (task_name, task_id, wait_time, dependent_task_ids) = task
        thread = myThread(task_id, task_name, wait_time)
        # Wait for dependent tasks to finish
        for dependent_task_id in dependent_task_ids:
            print("Task %d:  Waiting for task %d\n" % (task_id, dependent_task_id))
            threads[dependent_task_id - 1].join()
        thread.start()
        threads.append(thread)

    # Wait for all the tasks to finish
    for thread in threads:
        thread.join()

    print ("All tasks completed")

if __name__ == '__main__':
    main()