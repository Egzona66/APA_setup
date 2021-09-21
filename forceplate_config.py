from utils.constants import *


# Define a config class with all the options for data acquisition and post-hoc analysis
class Config:
    """
        ############## EXPERIMENT CONFIG  ####################
    """

    # ! ! ! ! ! !
    # ! fix this (arduino config) when the sensor is fixed:
    #     n_sensors = 3 # ! fix this because a sensor is broken

    # ! Change these for every recording
    experiment_folder = "F:\\Egzona\\2021\\160921_RED_F_1R_3" # ? This should be changed for everyexperiment to avoid overwriting 
    experiment_name = "160921_RED_F_1R_3"  # should be something like YYMMDD_MOUSEID, all files for an experiment will start with this name
    experiment_duration = 10*60  # acquisition duration in seconds, alternatively set as None

    # * Live video frames display and sensors data plotting
    live_display = False  # show the video frames as video is acquired
    live_plotting = False

    # * Check that these options are correct
    com_port = "COM4"  # port of the arduino running Firmata for data acquisition
    acquisition_framerate = 600  # fps of camera triggering -> NEED TO SPECIFY SLEEP TIME IN ARDUINO for frame triggering


    overwrite_files = True # ! ATTENTION: this is useful for debug but could lead to overwriting experimental data
    save_to_video = True  # ! decide if you want to save the videos or not

    """
        ############## LIVE SENSOR CONTROLS  ####################
    """
    # These commands control how to intepret the live read out to the sensors and act accordingly

    # Thresholds [When all th sensors have a readout > th, do something (e.g. open door)]
    live_sensors_ths = {"fr":.02, 
                        "fl":.02,
                        "hr":.02,
                        "hl":.02}
    time_on_sensors = 400 # mouse has to be on all sensors this number of ms before the door opens

    """
        ############## POST-HOC ANALYSIS  ####################
    """
    analysis_config = {
        "data_folder": "D:\\Egzona\\Plot-videos DLC", # where the data to analyse are stored
        "experiment_name": "140819_M1R2",
        "plot_colors": { "fr":magenta, 
                        "fl":blue, 
                        "hr":red, 
                        "hl":green},

        # * for composite video
        # ? run video_analysis.py
        "start_clip_time_s": None, # ? Create clips start at this point, in SECONDS
        "start_clip_time_frame": 1, # ? Create clips start at this point, in FRAMES
        "clip_n_frames": 100 , # duration of the clip in frames
        "clip_name":"test", 

        "outputdict":{ # for ffmpeg
                    # '-vcodec': 'mpeg4',  #  high fps low res
                    "-vcodec": "libx264",   #   low fps high res
                    '-crf': '0',
                    '-preset': 'slow',  # TODO check this
                    '-pix_fmt': 'yuvj444p',
                    "-framerate": "10", #   output video framerate 
                    # TODO this doesnt work FPS
                },


    }


    """
        ############## LIVE PLOTTING  ####################
    """
    live_plotting_config = {
        "n_timepoints": 100, # number of frames to show in the plot
    }



    """
        ############## CAMERA CONFIG  ####################
    """
    # * These options should not be changed frequently unless  something changes in the experiment set up

    camera_config = {
        "video_format": ".avi",
        "n_cameras": 1,  # changed on 05/11/2019 to run only one camera
        "timeout": 100,   # frame acquisition timeout

        # ? Trigger mode and acquisition options -> needed for constant framerate
        "trigger_mode": False,  # hardware triggering
        "acquisition": {    
            "exposure": "1000",
            "frame_width": "192",  # must be a multiple of 32
            "frame_height": "160", # must be a multiple of 32
            "gain": "12",
            "frame_offset_y": "672",
            "frame_offset_x": "704",
        },


        # all commands and options  https://gist.github.com/tayvano/6e2d456a9897f55025e25035478a3a50
        # pixel formats https://ffmpeg.org/pipermail/ffmpeg-devel/2007-May/035617.html

        # "outputdict":{ # for ffmpeg
        #     "-vcodec": "mpeg4",   #   low fps high res
        #     '-crf': '0',
        #     '-preset': 'slow',  # TODO check this
        #     '-pix_fmt': 'gray',  # yuvj444p
        #     "-cpu-used": "1",  # 0-8. defailt 1, higher leads to higher speed and lower quality
        #     # "-r": "100", #   output video framerate 
        #     "-flags":"gray",
        #     # "-ab":"0",
        #     # "-force_fps": "100"
        # },
        "outputdict":{
            "-c:v": 'libx264',   #   low fps high res
            "-crf": '17',
            "-preset": 'ultrafast',
            "-pix_fmt": 'yuv444p',
            "-r": str(acquisition_framerate),
        },

        "inputdict":{
            "-r": str(acquisition_framerate),
        }
    }

    """
        ############## FIRMATA ARDUINO CONFIG  ####################
    """
    # Arduino (FIRMATA) setup options
    # * Only change these if you change the configuration of the inputs coming onto the firmata arduino
    arduino_config = {
        "sensors_pins": {
            # Specify the pins receiving the input from the sensors
            "fr": 0, # Currently the inputs from the force sensors go to digital pins on the arduino board
            "fl": 2,
            "hr": 6, 
            "hl": 4,
        },
        "door_open_pin": 10,
        "tone_pin": 11,
        "door_status_pin": 8, # analog open
        "arduino_csv_headers": ["frame_number", "elapsed", "camera_timestamp", \
                            "fr", "fl", "hr", "hl",\
                            "tone_playing", "door_opening"],
        "sensors": [ "fr", "fl", "hr", "hl"],
    }

    sensors_names =  [ "fr", "fl", "hr", "hl"]
    n_sensors = 4 # ! fix this because a sensor is brocken

    def __init__(self): 
        return # don't need to do anything but we need this func
