# Neuropixels
Explanation of how the neuropixel/tracking data were analyzed.
How the data was acquired, the experimental setup and the data pre-processing pipeline are described in FC's thesis.

## Preprocessing
FC's experiements involved freeling moving mice. DLC was used to track multiple body parts ("body", "tail_base", "left_fl", "right_fl", "left_hl", "right_hl"). The DLC tracking data coordinates are loaded (`process_tracking_data`) and aligned across experiments (see Shamash 2021 and Claudi 2022) by registering them to a common template.Tracking data are saved at 60Hz sampling rate (video was at 60fps). To compare to spike data from the neuropizel probe, the tracking data were upsampled to 1000Hz. Each a "trace" (e.g. x-coordinates for the nose across frame) was sampled by interpolating across time-poitns uisng numpy's 1d interpolation functionality.

For each unit (neuron) from the neuropixel recording (and in the PRNc/PRNr) recorded during an experimental session we have the times in miliseconds at which each spike occurred (from the spike sorting software). With this, we create a binary array where each "bin" (entry) corresponds to a 1-ms interval and each bin has a value of either 0 or 1 depending on whether a spike occured during that bin. This binary array is convolved with a gaussian curve  (`calc_firing_rate`) to obtain a firing rate in units of spikes/s (firing rate is easier to use in analysis and smooths out a bit of time variability).

So in the end we have a set of arrays where each bin corresponds to a milliseconds in the recording (and they have length equal to the experiment's duration). These arrays store the tracking data for various body parts and the neurons' firing rates.