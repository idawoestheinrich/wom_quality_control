# Importing Libraries
import time 
from pathlib import Path
import logging
#from moku.instruments import Oscilloscope
#to convert data from arduino to array


"""
WOMqc acquisition and analysis class.

This class manages full data acquisition and processing for WOM scans,
including hardware control, waveform acquisition, integration, and
baseline correction.

Core responsibilities:
    - Hardware connection (Moku + Arduino)
    - 2D scan control (i, j positions)
    - Waveform acquisition and storage
    - Signal integration (PMT + SiPM channels)
    - Baseline correction pipeline
    - Real-time plotting (optional)
    - Data saving (CSV, binary, metadata)

Data structure:
    - waveforms: raw or baseline-corrected waveforms
    - integrated_data: integrated charge + metadata
    - baseline_wf: baseline-corrected waveforms
    - baseline_integrated_data: processed integrals

Modes:
    - dry_run: simulates full acquisition without hardware
    - plot_waveform: enables live waveform visualization
    - plot_heatmap: enables post-processing visualization

File handling:
    Automatically generates:
        - data folder
        - logfile
        - measurement outputs (csv/bin)

Attributes:
    device_ip, WOMname, ni, nj, rep
    pulsewidth settings, voltage ranges
    integration windows
    acquisition timing parameters
"""
class WOMqc:
    def __init__(self, device_ip, WOMname, ni, nj, rep, 
            pulsewidth_ch1, pulsewidth_ch2, voltage_range_PMT,  
            voltage_range_SiPM , rec_time_min, rec_time_max, 
            int_window_min_PMT, int_window_max_PMT, int_window_min_SiPM, 
            int_window_max_SiPM, frame_length_max, LEDamplitude = 1.9, 
            sipm_voltage = 40.7, heatup_roomtemp = False, heatupmin = False,  
            plot_waveform=False, plot_heatmap = False, PMsoff= False, 
            dry_run=False, date= None, load_existing=False):
    
        self.device_ip = device_ip  #IP adress of the Moku device
        self.WOMname = WOMname #characteristic WOM name
        self.ni = ni #number of positions in i direction
        self.nj = nj #number of positions in j direction
        self.rep  = rep #number of waveforms taken per position rep
        self.pulsewidth_ch1 = pulsewidth_ch1 #what should be the pulse length for the inner LED
        self.pulsewidth_ch2 = pulsewidth_ch2 #what should be the pulse length for the outer LED
        
        self.voltage_range_PMT = voltage_range_PMT #voltage range of the Moku input channels "4Vpp", "400mVpp", ...
        self.voltage_range_SiPM = voltage_range_SiPM #voltage range of the Moku input channels "4Vpp", "400mVpp", ...
       
        self.rec_time_min  = rec_time_min #record time window min
        self.rec_time_max = rec_time_max #record time window max
        self.int_window_min_PMT = int_window_min_PMT #integration window relativ to maximum
        self.int_window_max_PMT = int_window_max_PMT
        self.int_window_min_SiPM = int_window_min_SiPM #integration window relativ to maximum
        self.int_window_max_SiPM = int_window_max_SiPM
        self.frame_length_max = frame_length_max #number of bins in the frame 

        self.sipm_voltage = sipm_voltage
        self.LEDamplitude = LEDamplitude
        self.heatup_roomtemp = heatup_roomtemp
        self.heatupmin = heatupmin
        self.plot_waveform = plot_waveform #if true, plot example waveforms during the scan
        self.plot_heatmap = plot_heatmap #if true, plot heatmaps after the measurement

        self.dry_run = dry_run  #if true, no hardware is connected
        self.PMsoff = PMsoff

        self.date = date
        if not load_existing:
            self.foldername.mkdir(parents=True, exist_ok=True)
            self.logger = self._setup_logger()
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.i = 0
        self.j = 0

        self.device = None
        self.arduino = None

        self.waveforms = None
        self.integrated_data = None

        self.baseline_wf = None
        self.baseline_integrated_data = None
        self.baseline_integrated_data_eventwise = None

        

    @property    
    def filename(self):
        if self.date is None:
            self.date = time.strftime("%Y%m%d")#%H%M
        #return Path.cwd().parent / "data" / f"{date}_{self.WOMname}"
        return f"{self.date}_{self.WOMname}"
    
    @property    
    def foldername(self):
        #return Path.cwd().parent / "data" / f"{date}_{self.WOMname}"
        return Path("/Users/ida/Desktop/Research/ship/woms/quality_control_setup/wom_quality_control/data") / self.filename

    
    # Delegate hardware methods
    from ._hardware import connect_hardware, configure_scope, cleanup
    from ._hardware import write_read, motor_on,  motor_off, move_home, step_down, move_WOM_top, rotate_step

    #Acquisition Delegation Wrappers
    from ._acquisition import heatup, measurement_without_WOM, getdata, measure_position
    from ._acquisition import run_darkcount_scan, run_scan, run_longterm, run
  
    #Analysis Delegation Wrappers - Signal Processing
    from ._signal_processing import riemann_sum_peak, gaussian, EMG, correct_baseline_min, gain
    from ._signal_processing import create_baseline_data, grouped_mean_std, calculate_eventwise_ratios, apply_sipm_corrections
  
    # Data IO Delegation Wrappers - DataManager
    from ._data_io import _setup_logger, save_metadata, read_metadata, savecsv, readcsv, savebin, readbin, load
  
    #Vizualization Delegation Wrappers - Plotter
    from ._visualization import init_plotting, update_example_waveform, plot_temperature_over_time, plot_integral_over_time, heatmap_WOM, light_yield_1dim, charge_spectrum, plot_waveforms_position



    