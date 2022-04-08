# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np



FOLDER =  Path(r'E:\Egzona\Forceplate\analysis FEMALES')
FILE =  'Statistical_analysis_FP - FEMALES.xlsx'
SHEET = "24 BINS HL (2)"

if not FOLDER.exists():
    raise FileNotFoundError(f"Folder {FOLDER} does not exist")

if not (FOLDER / FILE).exists():
    import glob

    print(*glob.glob(str(FOLDER)+"/*"), sep="\n")
    raise FileNotFoundError(f"File '{FILE}' not found in folder '{FOLDER}'")

data = pd.read_excel(
  FOLDER / FILE, sheet_name=SHEET
)
print(data.head())

# %%
def plot_mean_and_error(y, yerr, ax, err_alpha=0.3, color="k", **kwargs):
    alpha = kwargs.pop("alpha", 1)
    lw = kwargs.pop("lw", 3)
    err_color = kwargs.pop("err_color", color)
    zorder = kwargs.pop("zorder", 90)
    x = kwargs.pop("x", np.arange(len(y)))

    ax.fill_between(
        x,
        y - yerr,
        y + yerr,
        alpha=err_alpha,
        zorder=zorder - 1,
        color=err_color,
    )
    ax.plot(x, y, alpha=alpha, lw=lw, zorder=zorder, color=color, **kwargs)


condition_one_mean = data["hl - CONTROL (no manipul) 27 trials"]
condition_one_var = data["AVG variance"]
condition_one_color = "green"
condition_one_name = "control"

condition_two_mean = data["hl - DTR (diphteria)  11 trials"]
condition_two_var = data["AVG variance.1"]
condition_two_color = "black"
condition_two_name = "DTR"

significant_column = "Unnamed: 17"  # use print(data.columns) to see al column names
significant = [s for s in data[significant_column].values if isinstance(s, str)]
significant = [1 if s == "SIGNIFICANT" else 0 for s in significant]



f, ax = plt.subplots(figsize=(16, 9))

plot_mean_and_error(
    condition_one_mean, 
    np.sqrt(condition_one_var), 
    ax, 
    color=condition_one_color,
    lw=4,
    label=condition_one_name
)

plot_mean_and_error(
    condition_two_mean, 
    np.sqrt(condition_two_var), 
    ax, 
    color=condition_two_color,
    lw=4,
    label=condition_two_name
)




yval = np.nanmax(np.vstack((condition_one_mean, condition_two_mean)))
yval += yval * .25
for bin, sig in enumerate(significant):
    if sig:
        ax.text(bin, yval, "*", fontsize=30, horizontalalignment="center", verticalalignment="center")
        ax.plot([bin, bin], [0, yval], lw=2, ls=":", color="k", alpha=.25, zorder=-1)

# for i in ange(len(significant)):
#     ax.plot([i, i], [0, yval], lw=2, ls=":", color="k", alpha=.25, zorder=-1)

ax.set(
    ylabel="body weight %",
    xlabel="bin number",
    xticks=np.arange(0, len(significant)),
    xticklabels=np.arange(0, len(significant)) + 1
)

ax.legend()

#        plot_mean_and_error(avg, std, main_axes[ch], color=dark_colors[ch], lw=4)


# %%
