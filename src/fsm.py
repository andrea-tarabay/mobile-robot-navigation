from threading import Thread

import random
import time

class Fsm(Thread):
    def __init__(self, update_callback):
        super().__init__(daemon=True)

        self.running = False
        self.update_callback = update_callback
        self.result = None

    def start(self):
        self.running = True
        if not self.is_alive():
            super().start()

    def stop_work(self):
        self.running = False

    def run(self):
        while self.running:
            for i in range(3):
                print(f"Thread running... {i+1}/5")           
                time.sleep(1)

            print("Thread completed!")
            self.result = random.randint(1, 100)  
            self.update_callback(self.result)

        print("Thread stopped.")