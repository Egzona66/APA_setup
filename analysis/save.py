import sys
import pandas as pd
import numpy as np
from loguru import logger

sys.path.append("./")


from analysis.process_data import DataProcessing
from analysis.fixtures import sensors

USE_COG = "CoG_centered"  # or CoG_centered


data = DataProcessing.reload()
save_fld = data.main_fld.parent
save_fld.mkdir(exist_ok=True)

logger.info(f'Saving data to: {save_fld}')

# Saving single trials
data.data.to_csv(save_fld / 'saved_all_trials.csv')

# Saving averages & COG
chs = sensors+['tot_weight']
averages = {ch:np.mean(np.vstack(data.data[ch].values), axis=0) for ch in chs}
stds = {ch+'_std':np.std(np.vstack(data.data[ch].values), axis=0) for ch in chs}
averages.update(stds)

COGs = np.dstack([v for v in data.data[USE_COG].values])
cog = np.mean(COGs, 2)
cog_std = np.std(COGs, 2)
averages['cog_x'] = cog[:, 0]
averages['cog_y'] = cog[:, 1]
averages['cog_std_x'] = cog[:, 0]
averages['cog_std_y'] = cog[:, 1]

pd.DataFrame(averages).to_csv(save_fld / 'saved_averages.csv')

