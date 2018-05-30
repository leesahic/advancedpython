
from threading import Thread
import time
import sys

exit_flag = 0

class myThread (Thread):

    def __init__(self, thread_id, thread_name, counter):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.counter = counter
      
    def run(self):
        sys.stdout.write("Starting " + self.thread_name + "\n")
        self.print_time(self.thread_name, 5, self.counter)
        sys.stdout.write("Exiting " + self.thread_name + "\n")

    def print_time(self, thread_name, counter, delay):
        while counter:
            if exit_flag:
                thread_name.exit()
            time.sleep(delay)
            print("  %s: %s\n" % (thread_name, time.ctime(time.time())))
            counter -= 1


def main():

    thread1 = myThread(1, "Thread-1", 1)
    thread2 = myThread(2, "Thread-2", 2)

    thread1.start()
    thread2.start()

    print("Exiting main thread...")

if __name__ == '__main__':
    main()

