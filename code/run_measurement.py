import WOMqc
#import importlib

#importlib.reload(WOMqc) 
WOMqc = WOMqc.WOMqc

scan = WOMqc(
    device_ip = "[fe80::7269:79ff:feb0:1098]", # Bitte anpassen! fe80::7269:79ff:feb0:1098%11 fe80::7269:79ff:feb0:109 
    WOMname = "WOM_FreiburgXT2mm_uncoated",#"WOM_FreiburgXT2mm_6_55mm_new_cover_thin_coating_wo_coupling",#testbeam_WOM_16_6_55mm_vertical_steps_4",
    #"WOM_FreiburgXT2mm_6_55mm_old_reflector_thin_coating_4_covered_gab_ring",
    ni = 26, #vertical step count - in longterm measurments of the number of iterations 
    nj = 20,#rotation step count 
    rep = 30,#repititons
    pulsewidth_ch1 = 75e-9, #77e-9
    pulsewidth_ch2 = 75e-9,
    voltage_range_PMT  = "400mVpp", 
    voltage_range_SiPM = "4Vpp", #"400mVpp" with no or uncoated WOM or "4Vpp" with coated WOM
    rec_time_min = 0.3e-6,
    rec_time_max = 0.7e-6,
    int_window_min_PMT = 20e-9, # 0.4e-6#relative to maximum      
    int_window_max_PMT = 40e-9, #0.55e-6 # relative to maximum  
    int_window_min_SiPM = 50e-9, # 0.4e-6#relative to maximum      
    int_window_max_SiPM = 200e-9, #0.55e-6 # relative to maximum  
    frame_length_max = 512,#256,
    heatup_roomtemp = 26.4,#if no heatup == False else number of minutes
    plot_waveform = False,
    plot_heatmap = True,
    dry_run = False,
    PMsoff= True # Photomultiplier off? True -> no filter on the saved waveforms - False-> abs(area) > 1e-10
    )
df = scan.run()
#df = scan.run_long