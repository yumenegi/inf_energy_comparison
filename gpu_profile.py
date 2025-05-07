import threading
import time
from py3nvml import py3nvml
import csv
import os
import run_inf_gpu
import plot_power_2
import csv

power_readings = []
time_stamps = []

# Stopping event
stop_profiling_event = threading.Event()

"""
Thread function to continuously profile GPU power usage.
"""
def gpu_profiler_thread(gpu_index=0, interval_ms=10):
    print("Profiler thread started.")
    try:
        py3nvml.nvmlInit()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        device_name = py3nvml.nvmlDeviceGetName(handle)
        print(f"Profiling power for {device_name.encode('utf-8')} (GPU {gpu_index})")

        interval_sec = interval_ms / 1000.0

        while not stop_profiling_event.is_set():
            try:
                power_mw = py3nvml.nvmlDeviceGetPowerUsage(handle)
                timestamp = time.time() # Record timestamp
                power_readings.append((timestamp, power_mw))

            except py3nvml.NVMLError as err:
                print(f"Error getting power: {err}")

            # Sleep for the interval or until the event is set
            stop_profiling_event.wait(interval_sec)

    except py3nvml.NVMLError as err:
        print(f"NVML Init Error in profiler thread: {err}")

    finally:
        try:
            py3nvml.nvmlShutdown()
            print("Profiler thread NVML shutdown.")
        except py3nvml.NVMLError as err:
             # Handle potential errors during shutdown if init failed
             print(f"NVML Shutdown Error in profiler thread: {err}")

    print("Profiler thread finished.")


def model_evaluation_thread(model, test_images, test_labels):
    """
    Thread function to run the model evaluation or inference workload.
    """
    print("Evaluation thread started.")
    time_stamps.append(time.time())
    run_inf_gpu.eval(model, test_images, test_labels)
    time_stamps.append(time.time())
    print("Model evaluation finished.")
    print("Total inference time is %.6f" % (time_stamps[1]-time_stamps[0]))

    # Sleeps thread for 10 seconds to let the GPU spindown and log the spindown
    # power profile
    time.sleep(10)
    print("Sleep done")


if __name__ == "__main__":
    # Initaialize model so the profiling period is only when the actual inference takes place
    model, test_images, test_labels = run_inf_gpu.init_model()

    # Create threads
    profiler = threading.Thread(target=gpu_profiler_thread, args=(0, 1)) # Profile GPU 0 every 10ms
    evaluator = threading.Thread(target=model_evaluation_thread, args=(model, test_images, test_labels))

    # Start the profiler thread
    profiler.start()
    print("Profiler thread started.")

    # Give the profiler a moment to initialize NVML and start the loop (optional but good practice)
    time.sleep(0.1)

    # Start the evaluation thread
    evaluator.start()
    print("Evaluation thread started.")

    # Wait for the evaluation thread to complete
    evaluator.join()
    print("Evaluation thread joined.")

    # Signal the profiler thread to stop
    stop_profiling_event.set()
    print("Stop event set for profiler.")

    # Wait for the profiler thread to complete
    profiler.join()
    print("Profiler thread joined.")

    # Now you have the power readings in the 'power_readings' list
    print(f"Collected {len(power_readings)} power readings.")

    # Samples
    print("\nSample Readings:")
    for i in range(min(5, len(power_readings))):
        ts, power_mw = power_readings[i]
        print(f"Timestamp: {ts:.4f}, Power: {power_mw} mW ({power_mw/1000.0:.2f} W)")

    if len(power_readings) > 10:
         print("...")
         for i in range(max(0, len(power_readings)-5), len(power_readings)):
            ts, power_mw = power_readings[i]
            print(f"Timestamp: {ts:.4f}, Power: {power_mw} mW ({power_mw/1000.0:.2f} W)")

    if power_readings:
        total_power_mw = sum(p for ts, p in power_readings)
        average_power_mw = total_power_mw / len(power_readings)
        print(f"\nAverage power during evaluation: {average_power_mw / 1000.0:.2f} W")

        with open("gpu_power_profile.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "power_mW"])
            writer.writerows(power_readings)
        print("Power readings saved to gpu_power_profile.csv")

        print("GPU started at %f and stopped at %f" % (time_stamps[0], time_stamps[1]))

    plot_power_2.plot_gpu_power_from_csv(csv_filepath="gpu_power_profile.csv", start_time = time_stamps[0], end_time = time_stamps[1])

    energy_j = plot_power_2.calculate_energy_from_csv("gpu_power_profile.csv", time_stamps[0], time_stamps[1])
    print("The energy used in total for the inference is %.3f Joules." % energy_j)
    print("The number of images is %i" % (len(test_labels)))
    print("The energy per inference is %.5f mJ" % (energy_j / len(test_labels) * 1000))