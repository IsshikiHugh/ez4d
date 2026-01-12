from .time import TimeMonitor, TimeMonitorDisabled

time_monitor = TimeMonitorDisabled()

def enable_time_monitor():
    global time_monitor
    time_monitor = TimeMonitor()

def disable_time_monitor():
    global time_monitor
    time_monitor = TimeMonitorDisabled()