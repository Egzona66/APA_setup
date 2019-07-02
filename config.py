
# Define a config class with all the options for data acquisition and post-hoc analysis


class Config:
    """
        ############## EXPERIMENT CONFIG  ####################
    """
    # * Change these for every recording
    experiment_folder = "E:\\Egzona"   # ? This should be changed for everyexperiment to avoid overwriting 
    experiment_name = "test_ground"  # should be something like YYMMDD_MOUSEID, all files for an experiment will start with this name

    # * Live video frames display and sensors data plotting
    live_display = True,  # show the video frames as video is acquired
    live_plotting = True

    # * Check that these options are correct
    com_port = "COM5"  # port of the arduino running Firmata for data acquisition
    acquisition_framerate = 100  # fps of camera triggering -> NEED TO SPECIFY SLEEP TIME IN ARDUINO for frame triggering


    overwrite_files = False # ! ATTENTION: this is useful for debug but could lead to overwriting experimental data
    acquisition_framerate = 100  # fps of camera triggering -> NEED TO SPECIFY SLEEP TIME IN ARDUINO for frame triggerin.
    # So this number is just indicative but the true acquisition rate depends on the trigger arduino


    """
        ############## POST-HOC ANALYSIS  ####################
    """
    analysis_config = {
        "data_folder": "E:\\Egzona", # where the data to analyse are stored
        "experiment_name": "test_ground",
        "plot_colors": { "fr":"m", 
                        "fl":"b", 
                        "hr":"g", 
                        "hl":"r"}
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
        "save_to_video": False,
        "video_format": ".mp4",
        "n_cameras": 2,
        "timeout": 100,   # frame acquisition timeout

        

        # ? Trigger mode and acquisition options -> needed for constant framerate
        "trigger_mode": True,  # hardware triggering
        "acquisition": {    
            "exposure": "5000",
            "frame_width": "1216",  # must be a multiple of 32
            "frame_height": "1024", # must be a multiple of 32
            "gain": "10",
            "frame_offset_y": "170",
        },

        "outputdict":{ # for ffmpeg
            '-vcodec': 'mpeg4',  #  high fps low res
            # "-vcodec": "libx264",   #   low fps high res
            '-crf': '0',
            '-preset': 'slow',  # TODO check this
            '-pix_fmt': 'yuvj444p',
            #"-framerate": "10000", #   output video framerate 
            # TODO this doesnt work FPS
        },
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
            "hr": 4, 
            "hl": 6,
        },
        "arduino_csv_headers": ["frame_number", "elapsed", "camera_timestamp", "fr", "fl", "hr", "hl"],
        "sensors": [ "fr", "fl", "hr", "hl"],
    }

    def __init__(self): 
        return # don't need to do anything but we need this func
