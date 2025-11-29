from tdmclient import ClientAsync, aw
from src.utils.repeated_timer import RepeatedTimer
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

class SpeedVarianceCalculator:
    def __init__(self, sampling_period=0.1, sampling_duration=30, left_motor_target=50, right_motor_target=50, strip_size_mm=50, strip_count=8):
        self.sampling_period = sampling_period
        self.sampling_duration = sampling_duration

        self.left_motor_target = left_motor_target
        self.right_motor_target = right_motor_target
        self.motor_covariance = None
        self.forward_speed_variance = None
        self.angular_speed_variance = None
        self.coef_thymio_speed_to_mm_s = None

        self.strip_size = strip_size_mm
        self.strip_count = strip_count

        self._peaks = []
        
        self.data = []


    def aquire_data(self, node, client: ClientAsync):
        aw(node.wait_for_variables()) # wait for Thymio variables values

        rtimer = RepeatedTimer(self.sampling_period, lambda: self.data.append({
                            "ground":list(node["prox.ground.reflected"]), 
                            "sensor":list(node["prox.ground.reflected"]),
                            "left_speed":node["motor.left.speed"],
                            "right_speed":node["motor.right.speed"]}))  # get data every 0.1s
        
        try:
            # time.sleep would not work here, use asynchronous client.sleep method instead
            aw(client.sleep(5))
            node.send_set_variables({
                "motor.left.target": [self.left_motor_target],
                "motor.right.target": [self.right_motor_target],
            })
            aw(client.sleep(self.sampling_duration - 5)) # your long-running job goes here...
        finally:
            rtimer.stop() # better in a try/finally block to make sure the program ends!
            node.send_set_variables({
                "motor.left.target": [0],
                "motor.right.target": [0],
            })

        return self.data
    
    def plot_ground_sensor_detection(self, threshold=500, distance=10):
        l_sensor = [x["ground"][0] for x in self.data]
        r_sensor = [x["ground"][1] for x in self.data]
        l_peaks = find_peaks(l_sensor, threshold, distance)[0]
        r_peaks = find_peaks(r_sensor, threshold, distance)[0]

        if len(l_peaks) == (self.strip_count - 1):
            self._peaks = l_peaks
        elif len(r_peaks) == (self.strip_count - 1):
            self._peaks = r_peaks
        else:
            print("Warning: peaks incorrectly detected, tune the threshold or distance parameters!")

        plt.plot(l_sensor, label="left sensor")
        plt.plot(r_sensor, label="right sensor")
        plt.plot(l_peaks, [l_sensor[idx] for idx in l_peaks], "o", label = "left sensor peaks")
        plt.plot(r_peaks, [r_sensor[idx] for idx in r_peaks], "o", label = "right sensor peaks")
        plt.xlabel("Time step [s]")
        plt.ylabel("Ground sensor measurement")
        plt.legend()

    def convert_thymio_speed_to_mm_s(self):
        if len(self._peaks) == 0:
            print("Error: No peaks detected, cannot compute speed conversion factor.")
            return None
        
        crossed_strip_count = len(self._peaks) - 1
        crossed_distance_mm = crossed_strip_count * self.strip_size

        crossing_time_s = self.sampling_period * (self._peaks[-1] - self._peaks[0])
        thymio_speed_mms = crossed_distance_mm/crossing_time_s

        self.coef_thymio_speed_to_mm_s = thymio_speed_mms / ((self.left_motor_target + self.right_motor_target)/2)

        return self.coef_thymio_speed_to_mm_s

    def compute_motor_speed_covariance(self):
        if self.coef_thymio_speed_to_mm_s is None:
            print("Error: Speed conversion factor not computed yet.")
            return None

        l_speed = [x["left_speed"] for x in self.data if (x["left_speed"] != 0 and x["left_speed"] != 65535)]
        r_speed = [x["right_speed"] for x in self.data if (x["right_speed"] != 0 and x["right_speed"] != 65535)]

        l_speed = np.array(l_speed) * self.coef_thymio_speed_to_mm_s
        r_speed = np.array(r_speed) * self.coef_thymio_speed_to_mm_s

        plt.plot(l_speed, label="Left motor")
        plt.plot(r_speed, label="Right motor")
        plt.plot((r_speed + l_speed)/2, label="Average motor speed", linestyle='--')
        plt.xlabel("Time step [s]")
        plt.ylabel("Measured Velocity [mm/s]")
        plt.legend()

        self.motor_covariance = np.cov(l_speed, r_speed)

        return self.motor_covariance
    
    def compute_speed_variance(self, axle_length=95):
        if self.motor_covariance is None:
            print("Error: Motor covariance not computed yet.")
            return None
        
        self.forward_speed_variance = (self.motor_covariance[0,0] + self.motor_covariance[1,1] + 2*self.motor_covariance[0,1]) / 4
        self.angular_speed_variance = (self.motor_covariance[0,0] + self.motor_covariance[1,1] - 2*self.motor_covariance[0,1]) / (axle_length**2)

        return self.forward_speed_variance, self.angular_speed_variance