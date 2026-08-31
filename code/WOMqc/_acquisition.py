# Importing Libraries
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import time 

#to convert data from arduino to array
import ast

'''
Handles data acqusition - coordinates the different steps in one measurement 
'''
"""
Perform hardware "heat-up" / stabilization routine.
This function drives both LED channels continuously for a fixed
time period to thermally and electrically stabilize the system
before measurement.
Steps:
    1. Configures output impedance (50 Ohm) for both channels
    2. Generates periodic pulse signals on:
        - Channel 1
        - Channel 2
    3. Runs continuous pulsing for 300 seconds
    4. Turns both LED outputs off
Purpose:
    - Stabilize LED response
    - Warm up electronics
    - Reduce drift in early measurements
Returns:
    None
"""
def heatup(self, deltaT_in = 7.25, deltaT_out = 8):
    self.logger.info("Heating LEDs to operating temperature.")
    if self.dry_run:
        return
    else:
        self.device.set_output_termination(channel=1, termination="50Ohm")#Set output configuartion 
        self.device.set_output_termination(channel=2, termination="50Ohm")#Set output configuartion 
        # Trigger on input Channel ch, rising edge, 1V 
        T_room = self.heatup_roomtemp
        response = self.write_read("1")
        self.logger.info(f"temperature {response}")  
        temp = ast.literal_eval(response)
        temp_in, temp_out = temp[0], temp[1]
        self.device.generate_waveform( 
            channel=1, type="Pulse", 
            amplitude=1.9, 
            frequency=1e6, 
            offset=0.95, 
            edge_time=2e-9, 
            pulse_width=self.pulsewidth_ch1,  
            phase=0) #inner LED
        self.device.generate_waveform(
            channel=2, type="Pulse", 
            amplitude=1.9, 
            frequency=1e6, 
            offset=0.95, 
            edge_time=2e-9, 
            pulse_width=self.pulsewidth_ch2,  
            phase=0.1) #outer LED                
        while temp_in <= T_room + deltaT_in: # Get PCB inside of the WOM to operational temperature
            response = self.write_read("1")
            self.logger.info(f"temperature {response}")  
            temp = ast.literal_eval(response)
            temp_in, temp_out = temp[0], temp[1]
            time.sleep(60) 
        self.device.generate_waveform( # swich to lower frequency to avoid overheating the inner LED
                channel=1, type="Pulse", 
                amplitude=1.9, 
                frequency=1e3, 
                offset=0.95, 
                edge_time=2e-9, 
                pulse_width=self.pulsewidth_ch1,    
                phase=0) 
        while temp_out <= T_room + deltaT_out: # Get outside PCB of the WOM to operational temperature
            response = self.write_read("1")
            self.logger.info(f"temperature {response}")  
            temp = ast.literal_eval(response)
            temp_in, temp_out = temp[0], temp[1]
            time.sleep(60)
        self.device.generate_waveform( # swich to lower frequency to avoid overheating the inner LED
                channel=2, type="Pulse", 
                amplitude=1.9, 
                frequency=1e3, 
                offset=0.95, 
                edge_time=2e-9, 
                pulse_width=self.pulsewidth_ch2,    
                phase=0.1)                 
        history_in = [temp_in]
        history_out = [temp_out]
        while True: 
                response = self.write_read("1")
                self.logger.info(f"temperature {response}")  
                temp = ast.literal_eval(response)
                temp_in, temp_out = temp[0], temp[1]
                history_in.append(temp_in)
                history_out.append(temp_out)
                if len(history_in) >= 30:
                    if (np.ptp(history_in[-30:])< 0.01) and (np.ptp(history_out[-30:])) < 0.02:
                        break
                time.sleep(3)    
        self.device.generate_waveform(channel=1, type="Off")#turn output (LED) off
        self.device.generate_waveform(channel=2, type="Off")#turn output (LED) off   
        return  

"""
Perform one measurement above the WOM for reference
"""     
def measurement_without_WOM(self):
    self.waveforms = []
    self.integrated_data = [] 
    if self.plot_waveform:
        self.init_plotting()
    try:
        self.logger.info(f"Measuring reference without WOM")
        waveforms, row = self.measure_position(999, 999)
        self.logger.info("Measurement finished") 
        self.waveforms.append(waveforms)
        self.integrated_data.append(row)
    except Exception as e:
        self.logger.error(f"Error at position j={j}, i={i}: {e}")
        self.waveforms = pd.DataFrame(self.waveforms)
        self.integrated_data = pd.DataFrame(self.integrated_data)
        return self.integrated_data, self.waveforms   # or break if you want to stop cleanly                
"""
Acquire waveform data and compute integrated signals for one LED channel.
This function performs repeated measurements for a single LED state
(ch = 1 or 2) and collects PMT + SiPM responses.
Workflow:
    1. Configures device trigger and LED pulse (unless dry-run)
    2. Repeats acquisition self.rep times:
        - Reads waveform data (real or simulated)
        - Integrates PMT and SiPM signals in predefined windows
        - Stores raw waveforms and per-shot integrals
        - Rejects empty pulses based on threshold
    3. Turns LED output off after acquisition
    4. Computes mean and standard deviation of integrals
Returns:
    tuple:
        PMT_data : np.ndarray
            Raw PMT waveforms
        SiPMin : np.ndarray
            Raw SiPM-in waveforms
        SiPMout : np.ndarray
            Raw SiPM-out waveforms
        PMT_mean : float
        PMT_std  : float
        SiPMin_mean : float
        SiPMin_std  : float
        SiPMout_mean : float
        SiPMout_std  : float
"""
def getdata(self, ch): #channel number ch, number of waveforms taken per position rep
    """
    Acquire data from PMT and the two SiPMs for one LED (ch) and and compute integrals.
    """
    int_data = np.zeros((3, self.rep))#Integrated waveforms for PMT, SiPMin, SiPMout
    data = np.zeros((3, self.rep, self.frame_length_max))#[[],[],[]]#Waveforms for PMT, SiPMin, SiPMout
    #datatemp = {} #temporary data storage
    output = (f"Output{ch}") #Set the output channel [(1, "in"), (2, "out")]
    #Set the pulse width for the LED 
    if ch == 1:
        pulsewidth =  self.pulsewidth_ch1 
    elif ch==2:
        pulsewidth = self.pulsewidth_ch2
    else:
        self.logger.warning("choose channel ", ch)
    if self.dry_run:
        self.logger.info(f"[DRY-RUN] Generating a waveform for Channel {ch}")
    else:
        self.device.set_output_termination(channel=ch, termination="50Ohm")#Set output configuartion 
        # Trigger on input Channel ch, rising edge, 1V      
        self.device.set_trigger(
                type="Edge", 
                level=1, 
                source=output, 
                edge="Rising")
            # Generate pulse on Channel ch -> LED emits a pulse of light 
        self.device.generate_waveform(
                channel=ch, type="Pulse", 
                amplitude=1.9, 
                frequency=1e3, 
                offset=0.95, 
                edge_time=2e-9, 
                pulse_width=pulsewidth,  
                phase=0) 
        time.sleep(0.25)
    count = 0
    stop = 0
    channels = ["ch1", "ch2", "ch3"] #PMT, SiPMin, SiPMout
    for count in range(self.rep): #measure rep times in one position 
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Returning a test waveforms for PMT and SiPMs")
            x = np.linspace(self.rec_time_min, self.rec_time_max, self.frame_length_max)
            mean = (self.rec_time_min + self.rec_time_max) / 2
            sigma = (self.rec_time_max-self.rec_time_min) / 10
            tau = (self.rec_time_max-self.rec_time_min) / 50
            height = 4
            datatemp = {
                "ch1": self.EMG(x, mean=mean, height=-height, sigma=sigma, tau=tau) 
                    + np.random.uniform(-0.02, 0.02, size=len(x)),
                "ch2": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                    + np.random.uniform(-0.02, 0.02, size=len(x)),
                "ch3": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                    + np.random.uniform(-0.02, 0.02, size=len(x)),
            }
        else:         
            datatemp = self.device.get_data()
        for i, channel in enumerate(channels):     
            #collecting the waveforms measured by the moku 
            if channel == "ch1":
                start, end, area = self.riemann_sum_peak(datatemp[channel], self.int_window_min_PMT, self.int_window_max_PMT)
                if self.PMsoff:
                    threshold = 0
                else:
                    if channel == "ch1":
                        threshold = 8e-11
                    else:  
                        threshold = 1e-10  
                if np.abs(area) >= threshold: #make sure that a waveform was integrated
                    int_data[i,count] = -area 
                else:
                    self.logger.info("No pulse, Area = ", -area, "count", count)
                    stop += 1
                    if stop >= 10:
                        self.logger.warning("No pulses in PMT detected after 10 attempts")
                        int_data[i,count] = -area 
                        break     
            else:   
                start, end, area = self.riemann_sum_peak(datatemp[channel], self.int_window_min_SiPM, self.int_window_max_SiPM) 
                #if channel == "ch2":
                #    area =- (self.int_window_min_SiPM + self.int_window_max_SiPM)*15e-3
                #if channel == "ch3":
                #    area =- (self.int_window_min_SiPM + self.int_window_max_SiPM)*20e-3
                if self.PMsoff:
                    threshold = 0
                else:
                    if channel == "ch1":
                        threshold = 8e-11
                    else:  
                        threshold = 1e-10                      
                if np.abs(area) >= threshold: #make sure that a waveform was integrated
                    if channel == "ch2":
                        int_data[i,count] = area - 0.014*(self.int_window_min_SiPM + self.int_window_max_SiPM)
                    if channel == "ch3":
                        int_data[i,count] = area - 0.021*(self.int_window_min_SiPM + self.int_window_max_SiPM)   
                else:
                    self.logger.info("No pulse, Area = ", np.abs(area), "count", count)
                    stop += 1
                    if stop >= 10:
                        self.logger.warning("No pulses in SiPMs detected after 10 attempts")
                        break  
            data[i,count] = datatemp[channel]                           
    if self.dry_run:
        self.logger.info(f"[DRY-RUN] Turning off output channel {ch}")
    else:  
        self.device.generate_waveform(channel=ch, type="Off")#turn output (LED) off       
    PMT_data, SiPMin, SiPMout = data[0], data[1], data[2]
    PMT_int, SiPMin_int, SiPMout_int = int_data[0], int_data[1], int_data[2]  
        #int_data = pd.DataFrame(int_data)
        #int_data.to_csv(dfname, sep=",")
    if ch == 1: #[(1, "in"), (2, "out")]
        PMT_ratio = PMT_int/SiPMin_int
        SiPM_ratio = SiPMout_int/SiPMin_int
        SiPM_ref = SiPMin_int
    elif ch == 2: 
        PMT_ratio = PMT_int/SiPMin_int
        SiPM_ratio = SiPMin_int/SiPMout_int
        SiPM_ref = SiPMout_int
    else:
        self.logger.warning("choose channel ", ch)        
    if self.rep > 1:
        return (
            PMT_data, #waveforms
            SiPMin, #waveforms
            SiPMout, #waveforms
            np.mean(PMT_ratio), np.std(PMT_ratio, ddof=1),  #mean of the integrals over rep waverforms and standarddeviation
            np.mean(SiPM_ratio), np.std(SiPM_ratio, ddof=1), #mean of the integrals over rep waverforms and standarddeviation
            np.mean(SiPM_ref), np.std(SiPM_ref, ddof=1) #mean of the integrals over rep waverforms and standarddeviation
        )
    elif self.rep == 1:
        return (
            PMT_data, #waveforms
            SiPMin, #waveforms
            SiPMout, #waveforms
            PMT_ratio[0], PMT_ratio[0],  #mean of the integrals over rep waverforms and standarddeviation
            SiPM_ratio[0], SiPM_ratio[0], #mean of the integrals over rep waverforms and standarddeviation
            SiPM_ref[0], SiPM_ref[0] #mean of the integrals over rep waverforms and standarddeviation
        )
    else: 
        self.logger.warning("choose rep > 0") 
"""
Measure a single scan position (j, i).
Acquires waveform data and integrated signals for both
LED states (in/out) at a fixed scan coordinate.
Workflow:
    1. Reads temperature from external controller
    2. Acquires waveforms via `getdata()` for:
        - LED in (ch=1)
        - LED out (ch=2)
    3. Computes:
        - Waveforms (raw traces)
        - Integrated charge per channel
        - Standard deviation of integrals
    4. Optionally updates live waveform plot
Takes:
    j : int
        Scan index in j-direction
    i : int
        Scan index in i-direction
Returns:
    tuple:
        waveforms : dict
            Raw waveform arrays per channel and LED state
        row : dict
            Integrated values, temperatures, and metadata
"""
def measure_position(self, j, i):
    row = {
        "j": j,
        "i": i,
        "time": time.strftime("%H:%M:%S")
    }
    waveforms = {
        "j": j,
        "i": i,  
    }
    if self.dry_run:
        temp = [20.0, 21.0]
    else:
        response = self.write_read("1")
        self.logger.info(f"temperature {response}")   
    #if self.dry_run == False:
        temp = ast.literal_eval(response)                    
        temp_in, temp_out = temp[0], temp[1]
            #self.logger.info(f"Temperature in={temp_in}, out={temp_out}")        
        row["temp_in"] = temp_in
        row["temp_out"] = temp_out         
    for ch, tag in [(1, "in"), (2, "out")]:
        (
            PMT_data, SiPMin, SiPMout,
            PMT_ratio, PMT_ratio_std, #PMT response
            SiPM_ratio, SiPM_ratio_std,  #SiPM_ratio, std - transmission
            SiPM_ref, SiPM_ref_std  #SiPM_ref, std - reference
        ) = self.getdata(ch) #get waveforms, their integrals and standard deviation
        if self.plot_waveform:
            self.update_example_waveform([np.array(PMT_data).mean(axis=0), np.array(SiPMin).mean(axis=0), np.array(SiPMout).mean(axis=0)], j, i, [f"PMT_{tag}", f"SiPMin_{tag}", f"SiPMout_{tag}"])
            time.sleep(0.1)
        waveforms[f"PMT_{tag}"] = PMT_data
        waveforms[f"SiPMin_{tag}"] = SiPMin
        waveforms[f"SiPMout_{tag}"] = SiPMout                        
        row[f"PMT_ratio_{tag}"] = PMT_ratio
        row[f"PMT_ratio_{tag}_std"] = PMT_ratio_std
        row[f"SiPM_ratio_{tag}"] = SiPM_ratio
        row[f"SiPM_ratio_{tag}_std"] = SiPM_ratio_std
        row[f"SiPM_ref_{tag}"] = SiPM_ref
        row[f"SiPM_ref_{tag}_std"] = SiPM_ref_std        
    #if self.dry_run == False:
    #    for _ in range(3):
    #        if a:   
    #            temp = ast.literal_eval(a)     
    #            break
    #        else: 
    #            a = self.write_read("1")
    #            time.sleep(0.05)
    #            self.logger.info("No temperature measured")                      
    return waveforms, row
'''
Full scan measurement
'''
"""
Run a darkcount or stability scan using LED pulsing and waveform acquisition.
This function performs automated data acquisition either in:
    - Darkcount mode (LED off, triggered reference pulse)
    - Stability mode (LED driven on selected channel)
Workflow:
    1. Configure device trigger and waveform generator
    2. Loop over scan positions (i-axis)
    3. Acquire repeated waveforms per position
    4. Compute integrated charge per waveform
    5. Store waveform and integrated data
    6. Optionally display live plots
    7. Convert results to pandas DataFrame
Data handling:
    - Raw waveforms stored per channel and LED state
    - Integrated charge stored per repetition and channel
    - Temperature is periodically queried from external controller
Returns:
    tuple:
        integrated_data (pd.DataFrame)
        waveforms (pd.DataFrame)
"""
def run_darkcount_scan(self, darkcount, external_trigger):
    self.waveforms = []
    self.integrated_data = []        
    if self.plot_waveform:
        self.init_plotting()
    dx = (self.rec_time_max - self.rec_time_min) / self.frame_length_max
    if darkcount:
            LEDs = [(1, "in")]
            self.logger.info(f"Measuring darkcounts, LED in")                
            self.device.set_output_termination(channel=3, termination="50Ohm")
            # Generate pulse on Channel ch\
            self.device.generate_waveform(channel=3, type="Pulse", amplitude=0.6, frequency=1e3, offset=0.3, edge_time=2e-9, pulse_width=100e-9,  phase=0)
            # Trigger on input Channel ch, rising edge, 1V 
            self.device.set_trigger(type="Edge", level=0.3, source="Output3", edge="Rising")                
    else:     
        LEDs = [(1, "in"), (2, "out")]
    for i in range(self.ni):
        row = {
                "j": 0,
                "i": i,#(ch-1)*self.ni+i,
                "time": time.strftime("%H:%M:%S")
            }
        waveforms = {
                "j": 0,
                "i": i,#(ch-1)*self.ni+i,  
            }            
        if self.dry_run:
                temp = [20.0, 21.0]
        else:
                response = self.write_read("1")
                self.logger.info(f"temperature {response}")   
        if i % 10 == 0: 
                self.logger.info(i)        
        for ch, tag in LEDs:
            output = (f"Output{ch}") 
            if ch == 1:
                pulsewidth =  self.pulsewidth_ch1 
            elif ch==2:
                pulsewidth = self.pulsewidth_ch2
            else:
                self.logger.warning("choose channel ", ch)
            if darkcount == False and external_trigger == False:
                self.logger.info(f"Measuring stability over time, LED {tag}")            
                self.device.set_output_termination(channel=ch, termination="50Ohm")
                # Generate pulse on Channel ch\
                self.device.generate_waveform(channel=ch, type="Pulse", amplitude=1.9, frequency=1e3, offset=0.95, edge_time=2e-9, pulse_width=pulsewidth,  phase=0)
                # Trigger on input Channel ch, rising edge, 1V 
                self.device.set_trigger(type="Edge", level=1, source=output, edge="Rising")       
            elif darkcount == False and external_trigger == True: 
                if ch == 2: 
                    self.logger.info(f"Measuring external trigger, LED out")   
                    self.device.set_trigger(type="Edge", level=0.7, source="External", edge="Rising")                        
                if ch == 1:    
                    self.logger.info(f"Measuring stability over time, LED {tag}")
                    self.device.set_output_termination(channel=ch, termination="50Ohm")
                    # Generate pulse on Channel ch\
                    self.device.generate_waveform(channel=ch, type="Pulse", amplitude=1.9, frequency=1e3, offset=0.95, edge_time=2e-9, pulse_width=pulsewidth,  phase=0)
                    # Trigger on input Channel ch, rising edge, 1V         
                    self.device.set_trigger(type="Edge", level=1, source=output, edge="Rising")                          
            stop = 0                
            channels = ["ch1", "ch2", "ch3"] #PMT, SiPMin 15mV offset, SiPMout 20mV offset
            int_data = np.zeros((3, self.rep))#Integrated waveforms for PMT, SiPMin, SiPMout
            data = np.zeros((3, self.rep, self.frame_length_max))#
            for count in range(self.rep): #measure rep times in one position 
                if self.dry_run:
                    self.logger.info(f"[DRY-RUN] Returning a test waveforms for PMT and SiPMs")
                    x = np.linspace(self.rec_time_min, self.rec_time_max, self.frame_length_max)
                    mean = (self.rec_time_min + self.rec_time_max) / 2
                    sigma = (self.rec_time_max-self.rec_time_min) / 10
                    tau = (self.rec_time_max-self.rec_time_min) / 50
                    height = 4
                    datatemp = {
                        "ch1": self.EMG(x,mean=mean,            height=-height, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                        "ch2": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                        "ch3": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                    }
                else:         
                    datatemp = self.device.get_data()
                for ch_idx, channel in enumerate(channels):     
                    #collecting the waveforms measured by the moku 
                    if darkcount:
                        area = np.sum(datatemp[channel]) * dx
                        int_data[ch_idx,count] = np.abs(area)              
                    else:                                            #collecting the waveforms measured by the moku  
                        start, end, area = self.riemann_sum_peak(datatemp[channel], self.int_window_min_SiPM, self.int_window_max_SiPM) 
                        if np.abs(area) > 5e-10: #make sure that a waveform was integrated
                                int_data[ch_idx,count] = np.abs(area)
                        else:
                                self.logger.info("No pulse, Area = ", area)
                                stop += 1
                                if stop >= 10:
                                    self.logger.warning("No pulses detected after 10 attempts")
                                    break     
                    data[ch_idx ,count] = datatemp[channel]    
            waveforms[f"PMT_{tag}"] = data[0]
            waveforms[f"SiPMin_{tag}"] = data[1]              
            waveforms[f"SiPMout_{tag}"] = data[2]              
            row[f"PMT_{tag}"] = int_data[0][0]           
            row[f"PMT_{tag}_std"] = 0           
            row[f"SiPMin_{tag}"] = int_data[1][0]         
            row[f"SiPMin_{tag}_std"] = 0         
            row[f"SiPMout_{tag}"] = int_data[2][0]          
            row[f"SiPMout_{tag}_std"] = 0  
                # other = "out" if tag == "in" else "in"   
                # waveforms[f"PMT_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[0]
                # waveforms[f"SiPMin_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[1]              
                # waveforms[f"SiPMout_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[2]              
                # row[f"PMT_{other}"] = np.zeros((3, self.rep))[0][0]           
                # row[f"PMT_std_{other}"] = 0           
                # row[f"SiPMin_{other}"] = np.zeros((3, self.rep))[1][0]         
                # row[f"SiPMin_std_{other}"] = 0         
                # row[f"SiPMout_{other}"] = np.zeros((3, self.rep))[2][0]          
                # row[f"SiPMout_std_{other}"] = 0  
            if self.plot_waveform:
                self.update_example_waveform([np.array(waveforms[f"PMT_{tag}"]).mean(axis=0), np.array(waveforms[f"SiPMin_{tag}"]).mean(axis=0), np.array(waveforms[f"SiPMout_{tag}"]).mean(axis=0)], 0, i, [f"PMT_{tag}", f"SiPMin_{tag}", f"SiPMout_{tag}"])
                time.sleep(0.1)
        temp = ast.literal_eval(response)
        temp_in, temp_out = temp[0], temp[1]
                #self.logger.info(f"Temperature in={temp_in}, out={temp_out}")
        row["temp_in"] = temp_in
        row["temp_out"] = temp_out
            #time.sleep(0.9)
        self.waveforms.append(waveforms)
        self.integrated_data.append(row)   
    if self.dry_run:
        self.logger.info(f"[DRY-RUN] Turning off output channel {ch}")
    else:  
        self.device.generate_waveform(channel=3, type="Off")#turn                
    if self.plot_waveform:
        plt.show()
        plt.ioff()
    self.waveforms = pd.DataFrame(self.waveforms)
    self.integrated_data = pd.DataFrame(self.integrated_data)
    return self.integrated_data, self.waveforms    
"""
Execute full WOM scan over a 2D grid of positions on the WOM.
Performs automated data acquisition across all (j, i) scan positions:
    - Moves system to each position
    - Measures waveforms and integrated signals
    - Stores results incrementally
    - Handles hardware stepping in both axes
Optional:
    - Real-time waveform plotting during acquisition
Error handling:
    - Logs failures at specific scan positions
    - Converts collected partial data to DataFrame
    - Stops scan safely and returns partial results
Scan flow:
    for each j row:
        for each i column:
            measure → store → step
        reset/rotate position
Returns:
    tuple:
        integrated_data (pd.DataFrame)
        waveforms (pd.DataFrame)
"""
def run_scan(self):
    if self.waveforms == None:
        self.waveforms = []
    if self.integrated_data == None:
        self.integrated_data = []
    if self.plot_waveform:
        self.init_plotting()
    for j in range(self.nj):
        if j > 0:
            response = self.rotate_step()
            self.logger.info(response)
        for i in range(self.ni):
            try:
                if i > 0:
                    response = self.step_down()
                    self.logger.info(response)
                self.logger.info(f"Measuring position j={j}, i={i}")
                waveforms, row = self.measure_position(j, i)
                self.logger.info("Measurement finished") 
                self.waveforms.append(waveforms)
                self.integrated_data.append(row)
                self.i = i                
            except Exception as e:
                self.logger.error(f"Error at position j={j}, i={i}: {e}")
                self.waveforms = pd.DataFrame(self.waveforms)
                self.integrated_data = pd.DataFrame(self.integrated_data)
                return self.integrated_data, self.waveforms   # or break if you want to stop cleanly
        response = self.move_WOM_top()
        self.logger.info(response)            
        self.j = j
    if self.plot_waveform:
        plt.show()
        plt.ioff()
    self.waveforms = pd.DataFrame(self.waveforms)
    self.integrated_data = pd.DataFrame(self.integrated_data)
    return self.integrated_data, self.waveforms
"""
Run full spatial WOM scan measurement.
Executes a 2D scan over (i, j) positions:
    1. Cleans up previous hardware state
    2. Saves metadata
    3. Connects and configures hardware
    4. Homes system and prepares scan position
    5. Runs full scan acquisition
    6. Saves results (CSV + binary)
    7. Optionally generates heatmaps
Error handling:
    - Logs exceptions
    - Attempts safe return to home position
    - Ensures cleanup in all cases
Returns:
    None
"""
def run(self):
    self.cleanup() 
    try:
        self.save_metadata()
        self.connect_hardware()
        self.logger.info(f"Connected Hardware")
        self.configure_scope()
        self.logger.info(f"Configured Hardware") 
        self.motor_on() 
        self.move_home()
        self.logger.info(f"Move Home")
        if self.heatup_roomtemp:
            self.heatup()    
        self.measurement_without_WOM()
        response = self.step_down()
        self.logger.info(response)
        self.logger.info(f"Move to WOM Top")
        time.sleep(1)
        self.run_scan()
        self.savecsv()
        self.savebin()
        self.motor_off() 
        if self.plot_heatmap:
            self.heatmap_WOM(ch1="PMT_ratio_in", ch2="PMT_ratio_in_std", baseline_subtract= False)
            self.heatmap_WOM(ch1="SiPM_ratio_in", ch2="SiPM_ratio_in_std", baseline_subtract= False)
            self.heatmap_WOM(ch1="SiPM_ref_in", ch2="SiPM_ref_in_std", baseline_subtract= False)
    except Exception as e:
        self.logger.error("Measurement failed:", e)
        self.motor_on() 
        self.move_home()
        self.motor_off()    
        self.cleanup()             
    finally:
        self.motor_on() 
        self.logger.info("Measurement completed successfully.")
        self.move_home()
        self.motor_off()   
        self.cleanup()


"""
Run long-term darkcount / stability measurement.
Executes a full automated acquisition sequence without spatial scanning:
    1. Cleans up previous hardware state
    2. Saves metadata
    3. Connects and configures hardware
    4. Runs darkcount or stability scan
    5. Saves results (CSV + binary)
    6. Optionally generates light-yield plots
Error handling:
    - Logs failures
    - Attempts safe return to home position
    - Ensures hardware cleanup in all cases
Modes:
    darkcount=True  → LED off reference measurement
    darkcount=False → stability measurement with LED on
Returns:
    None
"""
def run_longterm(self, darkcount = True):
    self.cleanup() 
    try:
        self.save_metadata()
        self.connect_hardware()
        self.logger.info(f"Connected Hardware")
        if darkcount:
            self.rec_time_min = 0
            self.rec_time_max = 1e-6
            self.frame_length_max = 1024
        if external_trigger:
            self.rec_time_min = 0.1e-6
            self.rec_time_max = 0.7e-6
            self.frame_length_max = 1024    
        self.configure_scope()
        self.logger.info(f"Configured Hardware")
        self.logger.info(f"Pulse LEDs for 20min")
        if self.heatup_roomtemp:
            self.heatup()    
        self.run_darkcount_scan(darkcount, external_trigger)
        self.savecsv()
        self.savebin()
        if self.plot_heatmap:
            self.light_yield_1dim()
            self.light_yield_1dim(ch1= "PMT_out", ch2 = "SiPMin_out", ch3="SiPMout_out")
    except Exception as e:
        self.logger.error("Measurement failed:", e)
        #self.move_home()
        self.cleanup()    
    finally:
        self.logger.info("Measurement completed successfully.")
        #self.move_home()
        self.cleanup() 

"""
Run a darkcount or stability scan using LED pulsing and waveform acquisition.
This function performs automated data acquisition either in:
    - Darkcount mode (LED off, triggered reference pulse)
    - Stability mode (LED driven on selected channel)
Workflow:
    1. Configure device trigger and waveform generator
    2. Loop over scan positions (i-axis)
    3. Acquire repeated waveforms per position
    4. Compute integrated charge per waveform
    5. Store waveform and integrated data
    6. Optionally display live plots
    7. Convert results to pandas DataFrame
Data handling:
    - Raw waveforms stored per channel and LED state
    - Integrated charge stored per repetition and channel
    - Temperature is periodically queried from external controller
Returns:
    tuple:
        integrated_data (pd.DataFrame)
        waveforms (pd.DataFrame)
"""
def run_darkcount_scan(self, darkcount, external_trigger):
    self.waveforms = []
    self.integrated_data = []        
    if self.plot_waveform:
        self.init_plotting()
    dx = (self.rec_time_max - self.rec_time_min) / self.frame_length_max
    if darkcount:
            LEDs = [(1, "in")]
            self.logger.info(f"Measuring darkcounts, LED in")                
            self.device.set_output_termination(channel=3, termination="50Ohm")
            # Generate pulse on Channel ch\
            self.device.generate_waveform(channel=3, type="Pulse", amplitude=0.6, frequency=1e3, offset=0.3, edge_time=2e-9, pulse_width=100e-9,  phase=0)
            # Trigger on input Channel ch, rising edge, 1V 
            self.device.set_trigger(type="Edge", level=0.3, source="Output3", edge="Rising")                
    else:     
        LEDs = [(1, "in"), (2, "out")]
    for i in range(self.ni):
        row = {
                "j": 0,
                "i": i,#(ch-1)*self.ni+i,
                "time": time.strftime("%H:%M:%S")
            }
        waveforms = {
                "j": 0,
                "i": i,#(ch-1)*self.ni+i,  
            }            
        if self.dry_run:
                temp = [20.0, 21.0]
        else:
                response = self.write_read("1")
                self.logger.info(f"temperature {response}")   
        if i % 10 == 0: 
                self.logger.info(i)        
        for ch, tag in LEDs:
            output = (f"Output{ch}") 
            if ch == 1:
                pulsewidth =  self.pulsewidth_ch1 
            elif ch==2:
                pulsewidth = self.pulsewidth_ch2
            else:
                self.logger.warning("choose channel ", ch)
            if darkcount == False and external_trigger == False:
                self.logger.info(f"Measuring stability over time, LED {tag}")            
                self.device.set_output_termination(channel=ch, termination="50Ohm")
                # Generate pulse on Channel ch\
                self.device.generate_waveform(channel=ch, type="Pulse", amplitude=1.9, frequency=1e3, offset=0.95, edge_time=2e-9, pulse_width=pulsewidth,  phase=0)
                # Trigger on input Channel ch, rising edge, 1V 
                self.device.set_trigger(type="Edge", level=1, source=output, edge="Rising")       
            elif darkcount == False and external_trigger == True: 
                if ch == 2: 
                    self.logger.info(f"Measuring external trigger, LED out")   
                    self.device.set_trigger(type="Edge", level=0.7, source="External", edge="Rising")                        
                if ch == 1:    
                    self.logger.info(f"Measuring stability over time, LED {tag}")
                    self.device.set_output_termination(channel=ch, termination="50Ohm")
                    # Generate pulse on Channel ch\
                    self.device.generate_waveform(channel=ch, type="Pulse", amplitude=1.9, frequency=1e3, offset=0.95, edge_time=2e-9, pulse_width=pulsewidth,  phase=0)
                    # Trigger on input Channel ch, rising edge, 1V         
                    self.device.set_trigger(type="Edge", level=1, source=output, edge="Rising")                          
            stop = 0                
            channels = ["ch1", "ch2", "ch3"] #PMT, SiPMin 15mV offset, SiPMout 20mV offset
            int_data = np.zeros((3, self.rep))#Integrated waveforms for PMT, SiPMin, SiPMout
            data = np.zeros((3, self.rep, self.frame_length_max))#
            for count in range(self.rep): #measure rep times in one position 
                if self.dry_run:
                    self.logger.info(f"[DRY-RUN] Returning a test waveforms for PMT and SiPMs")
                    x = np.linspace(self.rec_time_min, self.rec_time_max, self.frame_length_max)
                    mean = (self.rec_time_min + self.rec_time_max) / 2
                    sigma = (self.rec_time_max-self.rec_time_min) / 10
                    tau = (self.rec_time_max-self.rec_time_min) / 50
                    height = 4
                    datatemp = {
                        "ch1": self.EMG(x,mean=mean,            height=-height, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                        "ch2": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                        "ch3": self.EMG(x, mean=mean, height=height/2, sigma=sigma, tau=tau) 
                        + np.random.uniform(-0.02, 0.02, size=len(x)),
                    }
                else:         
                    datatemp = self.device.get_data()
                for ch_idx, channel in enumerate(channels):     
                    #collecting the waveforms measured by the moku 
                    if darkcount:
                        area = np.sum(datatemp[channel]) * dx
                        int_data[ch_idx,count] = np.abs(area)              
                    else:                                            #collecting the waveforms measured by the moku  
                        start, end, area = self.riemann_sum_peak(datatemp[channel], self.int_window_min_SiPM, self.int_window_max_SiPM) 
                        if np.abs(area) > 5e-10: #make sure that a waveform was integrated
                                int_data[ch_idx,count] = np.abs(area)
                        else:
                                self.logger.info("No pulse, Area = ", area)
                                stop += 1
                                if stop >= 10:
                                    self.logger.warning("No pulses detected after 10 attempts")
                                    break     
                    data[ch_idx ,count] = datatemp[channel]    
            waveforms[f"PMT_{tag}"] = data[0]
            waveforms[f"SiPMin_{tag}"] = data[1]              
            waveforms[f"SiPMout_{tag}"] = data[2]              
            row[f"PMT_{tag}"] = int_data[0][0]           
            row[f"PMT_{tag}_std"] = 0           
            row[f"SiPMin_{tag}"] = int_data[1][0]         
            row[f"SiPMin_{tag}_std"] = 0         
            row[f"SiPMout_{tag}"] = int_data[2][0]          
            row[f"SiPMout_{tag}_std"] = 0  
                # other = "out" if tag == "in" else "in"   
                # waveforms[f"PMT_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[0]
                # waveforms[f"SiPMin_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[1]              
                # waveforms[f"SiPMout_{other}"] = np.zeros((3, self.rep, self.frame_length_max))[2]              
                # row[f"PMT_{other}"] = np.zeros((3, self.rep))[0][0]           
                # row[f"PMT_std_{other}"] = 0           
                # row[f"SiPMin_{other}"] = np.zeros((3, self.rep))[1][0]         
                # row[f"SiPMin_std_{other}"] = 0         
                # row[f"SiPMout_{other}"] = np.zeros((3, self.rep))[2][0]          
                # row[f"SiPMout_std_{other}"] = 0  
            if self.plot_waveform:
                self.update_example_waveform([np.array(waveforms[f"PMT_{tag}"]).mean(axis=0), np.array(waveforms[f"SiPMin_{tag}"]).mean(axis=0), np.array(waveforms[f"SiPMout_{tag}"]).mean(axis=0)], 0, i, [f"PMT_{tag}", f"SiPMin_{tag}", f"SiPMout_{tag}"])
                time.sleep(0.1)
        temp = ast.literal_eval(response)
        temp_in, temp_out = temp[0], temp[1]
                #self.logger.info(f"Temperature in={temp_in}, out={temp_out}")
        row["temp_in"] = temp_in
        row["temp_out"] = temp_out
            #time.sleep(0.9)
        self.waveforms.append(waveforms)
        self.integrated_data.append(row)   
    if self.dry_run:
        self.logger.info(f"[DRY-RUN] Turning off output channel {ch}")
    else:  
        self.device.generate_waveform(channel=3, type="Off")#turn                
    if self.plot_waveform:
        plt.show()
        plt.ioff()
    self.waveforms = pd.DataFrame(self.waveforms)
    self.integrated_data = pd.DataFrame(self.integrated_data)
    return self.integrated_data, self.waveforms
