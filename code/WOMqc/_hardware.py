# Importing Libraries
import numpy as np
import serial 
import time 


try:
    from moku.instruments import Oscilloscope
except ImportError:
    Oscilloscope = None
#to convert data from arduino to array

'''
Handles low-level hardware connections and serial commands for 
the Moku Oscilloscope and Arduino stage controller.
'''
'''
├── connect_hardware()
├── configure_scope()
├── write_read()
├── motor_on()
├── motor_off()
├── move_home()
├── step_down()
├── move_WOM_top()
└── rotate_step()
├── cleanup()

'''
"""
    Connect to hardware devices (Moku + Arduino).    
    Initializes external instruments used for data acquisition:
        - Moku Oscilloscope (via IP connection)
        - Arduino serial interface (COM port)    
    In dry-run mode:
        Only logs connection attempts without hardware access.    
    Returns:
        None
"""
def connect_hardware(self):
        if self.dry_run:
            self.logger.debug(f"[DRY-RUN] connecting to Moku and Arduino")
        else:                
            self.device = Oscilloscope(ip=self.device_ip, force_connect=True)
            self.logger.debug(f"Connecting to Moku and Arduino")
            self.arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
"""
Configure oscilloscope input channels and acquisition settings.
Sets up Moku oscilloscope frontend for all detector channels:
    - Impedance and coupling
    - Voltage range per channel
    - Input source mapping
    - Timebase (recording window and resolution)
Channels:
    1 → PMT
    2 → SiPM in
    3 → SiPM out
In dry-run mode:
    Only logs configuration steps.
Returns:
    None
"""        
def configure_scope(self):
    if self.dry_run:
        self.logger.debug(f"[DRY-RUN] setting up Moku")
    else:
        self.device.set_frontend(
            channel=1, impedance="50Ohm",
            coupling="DC", range= self.voltage_range_PMT
        )
        self.device.set_frontend(
            channel=2, impedance="50Ohm",
            coupling="DC", range= self.voltage_range_SiPM
        )
        self.device.set_frontend(
            channel=3, impedance="50Ohm",
            coupling="DC", range=self.voltage_range_SiPM
        )
        self.device.set_source(channel=1, source="Input1")
        self.device.set_source(channel=2, source="Input2")
        self.device.set_source(channel=3, source="Input3")
        self.device.disable_input(channel=4)
        self.device.set_timebase(
            self.rec_time_min,
            self.rec_time_max,
            max_length = self.frame_length_max
        )
    """
    Comunication with Arduino
    command =
    1. Measure Temperature (inner & outer PCB)
    2. Rotate stepper motor 18 degrees forward
    3. Rotate stepper motor 18 degrees reverse
    4. Trigger Output 1
    5. Trigger Output 2
    6. Rotate stepper motor 180 degrees forward
    7. Rotate stepper motor 180 degress reverse
    8. Find magnet (1 full rotation max)")
    9. Rotate stepper motor 90 degrees forward
    10. Rotate stepper motor 90 degrees reverse
    0. Exit

    returns response from arduino
    """       
def write_read(self, command): 
    if self.dry_run:
        self.logger.debug(f"[DRY-RUN] Communicating with Arduino: sent {x}, received test data")
        return [20 + np.random.randn()*0.1, 21 + np.random.randn()*0.1]
    else:
        try:
            self.arduino.reset_input_buffer()  # clear old data
            self.arduino.write((command + '\n').encode())
            # Wait for response (up to timeout)
            start = time.time()
            while True:
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode(errors='ignore').strip()
                    return response
                if time.time() - start > 2:  # timeout (2 seconds)
                    return "[No response]"
        except Exception as e:
            return f"[Error: {e}]"

                    
'''
Motion control
'''
def motor_on(self):
    if self.dry_run:
        self.logger.debug(f"[DRY-RUN] turn motor on")
    else:
        response = self.write_read("11")
        time.sleep(1)
        self.logger.info(response)            
def motor_off(self):
    if self.dry_run:
        self.logger.debug(f"[DRY-RUN] turn motor off")
    else:
        response = self.write_read("12")
        time.sleep(1) 
        self.logger.info(response)            
"""
Move system to home position.
In dry-run mode, only logs the action.
Otherwise:
    - Sends home command multiple times
    - Rotates back by current j-position steps
    - Ensures mechanical reset of angular axis
Returns:
    None
"""
def move_home(self):
    if self.dry_run:
        self.logger.debug(f"[DRY-RUN] move to home position")
    else:
        for _ in range(5):
            response = self.write_read("5")
            time.sleep(1)
            self.logger.info(response)                
        #rotate to magnet
        if self.j <= 4:
            for _ in range(4):
                response = self.write_read("2")
                self.logger.info(response)
            response = self.write_read("8")
            time.sleep(1)
            self.logger.info(response)
        else:
            response = self.write_read("8")
            time.sleep(1)
            self.logger.info(response)  
"""
Move system to home position.
In dry-run mode, only logs the action.
Otherwise:
    - Sends home command multiple times
    - Rotates back by current j-position steps
    - Ensures mechanical reset of angular axis
Returns:
    Response message from Arduino after moving step down
"""
def step_down(self):
    if self.dry_run:
        return "[DRY-RUN] move step down"
    else:
        response = self.write_read("4")
        time.sleep(0.3)
    return response 
"""
Move system to the top (WOM reference position).
In dry-run mode, only logs the action.
Otherwise:
    - Sends repeated home/top commands
    - Waits for mechanical stabilization
Returns:
    Message indicating completion of move to WOM top
"""                        
def move_WOM_top(self):
    if self.dry_run:
        return "[DRY-RUN] move from bottom to WOM top"
    else:
        for _ in range(2):
            response = self.write_read("5")
            time.sleep(1) 
            self.logger.info(response)
        return "Move to WOM top"  
"""
Rotate system by one angular step.
In dry-run mode, only logs the action.
Otherwise:
    - Triggers rotation command (currently commented or pending)
    - Waits for motion completion
Returns:
    Response message from Arduino after rotating step
"""
def rotate_step(self):
    if self.dry_run:
        return "[DRY-RUN] rotate step"
    else: 
        response = self.write_read("2")
        time.sleep(0.3)
        return(response)   

def cleanup(self):
    if self.device:
        self.device.relinquish_ownership()
        self.logger.info("Disconnect Moku")
    if self.arduino:
        self.arduino.close()
        self.logger.info("Disconnect Arduino")