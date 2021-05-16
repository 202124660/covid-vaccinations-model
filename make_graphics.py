from model import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as ticker
import ast
from datetime import date

make_covid_papers = True
make_real_doses_data_graph = True
make_doses_capacity_example = True
make_vaccine_comparison = True
make_contact_matrix_heatmap = True
make_heatmap = False
format_heatmap = True
make_vax_start_date_cum = True
make_vax_start_date_daily = True
make_vax_start_date_samegraph = True
make_vax_per_day_daily = True
make_vax_per_day_cum = True
make_vax_per_day_samegraph = True
sns.set(style="whitegrid", rc={'figure.figsize': (10, 6.25)})
palette_five = ['#0f0757', '#940061', '#e83344', '#f06f2b', '#ffcc00']
palette_neutral = [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                   (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
                   (0.23299120924703914, 0.639586552066035, 0.9260706093977744)]
textwidth = 418.25368
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10.95,
    "font.size": 10.95,
    'axes.titlesize': 10.95,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width
    fig_width_pt_fn = width_pt * fraction
    inches_per_pt_fn = 1 / 72.27
    golden_ratio_fn = (5 ** .5 - 1) / 2
    fig_width_in_fn = fig_width_pt_fn * inches_per_pt_fn
    fig_height_in_fn = fig_width_in_fn * golden_ratio_fn * (subplots[0] / subplots[1])
    return fig_width_in_fn, fig_height_in_fn


def run_model(I0_fn=1, Rval_fn=1.1, priority_fn=None, use_priority_fn=True, vdelay_fn=22, vprob_fn=0.76,
              finalvprob_fn=0.813, vdelay2_fn=14, vfromday_fn=150, vperday_fn=100000, dosegap_fn=84,
              vaxcapacityincrease_fn=1000.0, vaxmax_fn=500000):
    TotPop_fn = np.dot(1e6, [3.737, 4.136, 4.128, 3.745, 4.057, 4.437, 4.581, 4.434, 4.190, 4.182, 4.604, 4.578, 3.975,
                             3.395, 3.388, 2.527, 1.733, 1.704])
    Tmax_fn = 1000
    Kf_fn = [0.000007, 0.000012, 0.000022, 0.000041, 0.000075, 0.000137, 0.000252, 0.000461, 0.000845, 0.001549,
             0.002838, 0.0052, 0.009528, 0.017458, 0.031989, 0.058614, 0.107399, 0.283009]
    nP_fn = 100
    ndays_fn = Tmax_fn
    wave_fn = 1
    delay_fn = 14
    pop_fn = PopDyn(TotPop_fn, I0_fn, Tmax_fn, Rval_fn, Kf_fn, nP_fn, wave_fn, contact_matrix, priority_fn,
                    use_priority_fn)
    pop_fn.intialise(wave_fn, delay_fn)
    pop_fn.iterate(ndays_fn, vdelay_fn, vprob_fn, vfromday_fn, vperday_fn, dosegap_fn, finalvprob_fn, vdelay2_fn,
                   vaxcapacityincrease_fn, vaxmax_fn)
    return pop_fn.S, pop_fn.I, pop_fn.R, pop_fn.Fat, pop_fn.vaccinated


def shading(ax_fn, numlines, alpha, palette, legend, xlim, loc=False, commas=True, linewidth=4.0, dashed=False,
            indivlinestyles=False):
    lines_list = []
    for i in range(numlines):
        lines_list.append(ax_fn.lines[i])
    x = []
    y = []
    for i in range(numlines):
        x.append(lines_list[i].get_xydata()[:, 0])
        y.append(lines_list[i].get_xydata()[:, 1])
    for i in range(numlines):
        ax_fn.fill_between(x[i], y[i], color=palette[i], alpha=alpha)
    ax_fn.set_frame_on(False)
    if xlim:
        ax_fn.set_xlim(right=xlim)
    if type(dashed) is list:
        for i in range(len(dashed)):
            if dashed[i]:
                ax_fn.lines[i].set_linestyle("--")
    elif dashed:
        for line in ax_fn.lines:
            line.set_linestyle("--")
    if indivlinestyles:
        for num, line in enumerate(ax_fn.lines[:numlines]):
            line.set_linestyle(indivlinestyles[num])
    if legend:
        if not loc:
            leg = ax_fn.legend()
        else:
            leg = ax_fn.legend(loc=loc)
        for num, line in enumerate(leg.get_lines()):
            line.set_linewidth(linewidth)
            if type(dashed) is list:
                if dashed[num]:
                    line.set_linestyle("--")
            if indivlinestyles:
                line.set_linestyle(indivlinestyles[num])
    if commas:
        ax_fn.get_yaxis().set_major_formatter(
            ticker.FuncFormatter(lambda x_lambda, p_lambda: format(int(x_lambda), ',')))


def cumulative_to_daily(var1, var2="", absol=False):
    nvar1 = []
    nvar2 = []
    count = 0
    for i in var1[1:]:
        if absol:
            toadd = abs(i - var1[count])
        else:
            toadd = i - var1[count]
        if toadd < 0.01:
            toadd = 0
        nvar1.append(toadd)
        count += 1
    if var2 != "":
        count = 0
        for i in var2[1:]:
            if absol:
                toadd = abs(i - var2[count])
            else:
                toadd = i - var2[count]
            if toadd < 0.01:
                toadd = 0
            nvar2.append(toadd)
            count += 1
        return nvar1, nvar2
    else:
        return nvar1


TotPop = np.dot(1e6, [3.737, 4.136, 4.128, 3.745, 4.057, 4.437, 4.581, 4.434, 4.190, 4.182, 4.604, 4.578, 3.975, 3.395,
                      3.388, 2.527, 1.733, 1.704])
Tmax = 1000
asize = Tmax + 1
Kf = [0.000007, 0.000012, 0.000022, 0.000041, 0.000075, 0.000137, 0.000252, 0.000461, 0.000845, 0.001549, 0.002838,
      0.0052, 0.009528, 0.017458, 0.031989, 0.058614, 0.107399, 0.283009]
nP = 100
ndays = Tmax
wave = 1
delay = 14
I0 = 1
Rval = 1.1
contact_matrix = np.loadtxt(open("contact_matrix.csv", "rb"), delimiter=",") * 18 / 400
# priority = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # young -> old
# priority = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] # old -> young
priority = [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]  # cmx
use_priority = True
vdelay = 14  # pfizer delay until fully effective after 1 dose
vprob = 0.926  # pfizer effectiveness after 1 dose
finalvprob = 0.95  # pfizer effectiveness after 2 doses
vdelay2 = 7  # pfizer delay until fully effective after 2 doses
vfromday = 150  # how long to wait until vaccinations start
vperday = 100000  # how many vaccines per day
dosegap = 84  # gap between doses
vaxcapacityincrease = vperday / 100
vaxmax = vperday * 5

if make_vaccine_comparison:
    plt.figure(figsize=set_size(textwidth))
    sns.set_palette(palette_neutral)
    plt.xlabel("Number of days after first dose")
    plt.ylabel("Average level of immunity")
    nPad = 101
    vdelayb = 22  # az delay until fully effective after 1 dose
    vprobb = 0.76  # az efficacy after 1 dose
    finalvprobb = 0.813  # az efficacy after 2 doses
    vdelay2b = 14  # az delay until fully effective after 2 doses
    dosegapb = 84  # gap between doses
    vdelayc = 14  # mod delay until fully effective after 1 dose
    vprobc = 0.921  # mod efficacy after 1 dose
    finalvprobc = 0.941  # mod efficacy after 2 doses
    vdelay2c = 14  # mod delay until fully effective after 2 doses
    dosegapc = 21  # gap between doses
    df_linear = pd.DataFrame(np.column_stack([100 - 100 * np.array(
        [1 - (vprob * i / (vdelay - 1)) for i in range(vdelay)] + [1 - vprob] * (dosegap - vdelay) + [
            1 - vprob - ((finalvprob - vprob) * i / (vdelay2 - 1)) for i in range(vdelay2)] + [1 - finalvprob] * (
                nPad - dosegap - vdelay2)), 100 - 100 * np.array(
        [1 - (vprobc * i / (vdelayc - 1)) for i in range(vdelayc)] + [1 - vprobc] * (dosegapc - vdelayc) + [
            1 - vprobc - ((finalvprobc - vprobc) * i / (vdelay2c - 1)) for i in range(vdelay2c)] + [1 - finalvprobc] * (
                nPad - dosegapc - vdelay2c)), 100 - 100 * np.array(
        [1 - (vprobb * i / (vdelayb - 1)) for i in range(vdelayb)] + [1 - vprobb] * (dosegapb - vdelayb) + [
            1 - vprobb - ((finalvprobb - vprobb) * i / (vdelay2b - 1)) for i in range(vdelay2b)] + [1 - finalvprobb] * (
                nPad - dosegapb - vdelay2b))]), columns=["Pfizer-BioNTech", "Moderna", "Oxford-AstraZeneca"])
    shading(sns.lineplot(data=df_linear, dashes=False, linewidth=2), 3, 0.1, palette_neutral, True, 100, commas=False,
            linewidth=2, indivlinestyles=["-", "--", "-."])
    plt.ylim(-5, 100)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.tick_params(axis="y", direction="out", pad=-4)
    ax.tick_params(axis="x", direction="out", pad=-2)
    plt.savefig('graphics/vax-comparison.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

vdelay = 22  # az delay until fully effective after 1 dose
vprob = 0.76  # az efficacy after 1 dose
finalvprob = 0.813  # az efficacy after 2 doses
vdelay2 = 14  # az delay until fully effective after 2 doses

if make_contact_matrix_heatmap:
    cmx_columns = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                   "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
    cmx_columnsx = ["0-4", "5-9", " 10-14", " 15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                    "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
    cmx_df = pd.DataFrame(data=contact_matrix, columns=cmx_columnsx, index=cmx_columns)
    sns.heatmap(cmx_df, cmap="rocket_r", square=True, vmin=0, vmax=0.7, rasterized=True)
    ax = plt.gca()
    plt.xlim(0, 18)
    plt.ylim(0, 18)
    plt.xlabel("Age group")
    plt.ylabel("Age group")
    plt.savefig('graphics/contact-matrix.pdf', dpi=1200, bbox_inches="tight", transparent=True)
    plt.show()

if make_covid_papers:
    plt.figure(figsize=set_size(textwidth))
    covid_papers = np.loadtxt(open("frequency-covid-papers-nodates2.csv", "rb"), delimiter=",")
    cov_papers_df = pd.DataFrame(data=covid_papers, columns=["date", "frequency"])
    times = pd.date_range("2020-01-10", periods=len(cov_papers_df["frequency"]), freq="1D")
    cov_papers_df["date"] = times
    cov_papers_freqs = np.array(cov_papers_df["frequency"])
    cov_papers_df["frequency"] = cov_papers_df["frequency"].cumsum()
    plt.xlim(date(2020, 1, 10), date(2021, 5, 1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'\textrm{%d %b %Y}'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=53))
    shading(
        sns.lineplot(x="date", y="frequency", data=cov_papers_df, dashes=False, linewidth=2, color=palette_neutral[2]),
        1, 0.1, palette_neutral[2:], False, False, commas=True, linewidth=2)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Number of COVID-19 papers released")
    plt.gca().tick_params(axis="y", direction="out", pad=-3)
    plt.gca().tick_params(axis="x", direction="out", pad=-3)
    plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
    plt.savefig('graphics/covid-papers.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

if make_real_doses_data_graph:
    sns.set_palette([palette_neutral[1], palette_neutral[0]])
    plt.figure(figsize=set_size(textwidth))
    doses_csv = np.loadtxt(open("real-vax-doses-data-nodates.csv", "rb"), delimiter=",")
    vax_doses_data_df = pd.DataFrame(data=doses_csv, columns=["Date", "First dose", "Second dose"])
    times = pd.date_range("2021-01-14", periods=len(vax_doses_data_df["First dose"]), freq="1D")
    vax_doses_data_df["Date"] = times
    plt.xlim(date(2021, 1, 13), date(2021, 5, 9))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'\textrm{%d %b %Y}'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.stackplot(vax_doses_data_df["Date"], vax_doses_data_df["First dose"], vax_doses_data_df["Second dose"],
                  edgecolor="none", alpha=0.8, labels=["First dose (actual)", "Second dose (actual)"])
    plt.plot(vax_doses_data_df["Date"], [230000 + i * 2300 for i in range(len(times) + 7)][7:], "--", color="royalblue",
             label="Both doses (modelled)")
    plt.plot(vax_doses_data_df["Date"],
             list([230000 + i * 2300 for i in range(84)] + [64 * 2300 for i in range(len(times) + 7 - 84)])[7:], "k--",
             label="First dose (modelled)")
    plt.gcf().autofmt_xdate()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    ax.legend(handles, labels, loc="lower left")
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Number of vaccinations")
    plt.gca().tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
    plt.gca().tick_params(axis="y", direction="out", pad=-3)
    plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
    plt.savefig('graphics/doses-real-data.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

if make_doses_capacity_example:
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(set_size(textwidth, fraction=0.9, subplots=(2, 1)))
    sns.set_palette([palette_neutral[1], palette_neutral[0]])
    pop = PopDyn(TotPop, I0, Tmax, Rval, Kf, nP, wave, contact_matrix, priority, use_priority, optimise=False)
    pop.intialise(wave, delay)
    vperday = 100000
    vaxcapacityincrease = 0
    fname = "test{}.pdf".format("0")
    pop.iterate(ndays, vdelay, vprob, vfromday, vperday, dosegap, finalvprob, vdelay2, vaxcapacityincrease, vaxmax)
    pop.create_doses_lists(dosegap)
    df_vax_doses = pd.DataFrame(np.column_stack([pop.vch[150:], pop.vch2[150:-84]]),
                                columns=["First dose", "Second dose"])
    ax[0].stackplot([i for i in range(len(pop.vch[150:]))], pop.vch[150:], pop.vch2[150:-84], edgecolor="none",
                    alpha=0.8)
    ax[0].set_xlim(-5, 300)
    ax[0].set_ylim(top=410000)
    ax[0].set_title("(a) Daily vaccination capacity remaining constant.")
    ax[0].legend(["First dose", "Second dose"], loc="upper left")
    ax[0].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
    for spine in ax[0].spines.values():
        spine.set_visible(False)
    ax[0].tick_params(axis="y", direction="out", pad=-3)
    vaxcapacityincrease = vperday / 100
    vaxmax = vperday * 5
    pop.iterate(ndays, vdelay, vprob, vfromday, vperday, dosegap, finalvprob, vdelay2, vaxcapacityincrease, vaxmax)
    pop.create_doses_lists(dosegap)
    df_vax_doses = pd.DataFrame(np.column_stack([pop.vch[150:], pop.vch2[150:-84]]),
                                columns=["First dose", "Second dose"])
    ax[1].stackplot([i for i in range(len(pop.vch[150:]))], pop.vch[150:], pop.vch2[150:-84], edgecolor="none",
                    alpha=0.8)
    ax[1].set_xlim(-5, 300)
    ax[1].set_ylim(top=410000)
    ax[1].set_title("(b) Daily vaccination capacity increasing by 1,000 per day.")
    ax[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
    for spine in ax[1].spines.values():
        spine.set_visible(False)
    ax[1].set_xlabel("Days")
    ax[1].tick_params(axis="y", direction="out", pad=-3)
    fig.supylabel('Number of vaccinations per day', fontsize=10.95)
    fig.supxlabel('')
    ax[0].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
    ax[1].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
    ax[0].tick_params(axis="y", which="major", bottom=True, color="lightgray", width=1)
    ax[1].tick_params(axis="y", which="major", bottom=True, color="lightgray", width=1)
    fig.savefig('graphics/num-each-dose-per-day-both.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# generate data for heatmap
categories = ["Start date", "Uniform strategy", "Age-based strategy", "Contact-based strategy"]
numruns = 251
numrunsperday = 350
numrunsperdaystarting = 1
vax_matrix = np.zeros((numrunsperday, numruns))
vax_matrix_worst = np.zeros((numrunsperday, numruns))
if make_heatmap:
    counter = 0
    percs = [i / 10 for i in range(1001)]
    perccounter = 0
    text_file = open("heatmap-data{}-{}.txt".format(str(numrunsperdaystarting), str(numrunsperday)), "w")
    starttime = timer()
    maxrate = 999999999999
    oldelapsed = 0
    oldnumruns = 0
    for j in range(numrunsperdaystarting, numrunsperday + 1):
        noprio = []
        oldyoung = []
        cmxprio = []
        xvals = []
        for i in range(numruns):
            if (100 * ((j - numrunsperdaystarting) * numruns + i) / (
                    numruns * (numrunsperday - numrunsperdaystarting + 1))) >= percs[perccounter]:
                elapsed = timer() - starttime
                hours = round((elapsed / 60) // 60)
                mins = round((elapsed / 60) % 60)
                proportiondone = ((j - numrunsperdaystarting) * numruns + i) / (
                        numruns * (numrunsperday - numrunsperdaystarting + 1))
                if proportiondone == 0:
                    print("{}% done ({}/{}).".format(round(100 * proportiondone, 2),
                                                     3 * ((j - numrunsperdaystarting) * numruns + i),
                                                     3 * numruns * (numrunsperday - numrunsperdaystarting + 1)))
                else:
                    if elapsed - oldelapsed < maxrate:
                        maxrate = elapsed - oldelapsed
                        print(
                            "{}% done ({}/{}). {}h {}m elapsed, {}h {}m left. Running at {}% max speed (new record!), {}s per run.".format(
                                round(100 * proportiondone, 2), 3 * ((j - numrunsperdaystarting) * numruns + i),
                                                                3 * numruns * (
                                                                        numrunsperday - numrunsperdaystarting + 1),
                                round((elapsed / 60) // 60),
                                round((elapsed / 60) % 60), round(((elapsed / proportiondone - elapsed) / 60) // 60),
                                round(((elapsed / proportiondone - elapsed) / 60) % 60),
                                round(100 * maxrate / (elapsed - oldelapsed), 2),
                                round((elapsed - oldelapsed) / ((j - numrunsperdaystarting) * numruns + i - oldnumruns),
                                      2)))
                    else:
                        print(
                            "{}% done ({}/{}). {}h {}m elapsed, {}h {}m left. Running at {}% max speed, {}s per run.".format(
                                round(100 * proportiondone, 2), 3 * ((j - numrunsperdaystarting) * numruns + i),
                                                                3 * numruns * (
                                                                        numrunsperday - numrunsperdaystarting + 1),
                                round((elapsed / 60) // 60),
                                round((elapsed / 60) % 60), round(((elapsed / proportiondone - elapsed) / 60) // 60),
                                round(((elapsed / proportiondone - elapsed) / 60) % 60),
                                round(100 * maxrate / (elapsed - oldelapsed), 2),
                                round((elapsed - oldelapsed) / ((j - numrunsperdaystarting) * numruns + i - oldnumruns),
                                      2)))
                oldnumruns = (j - numrunsperdaystarting) * numruns + i
                oldelapsed = elapsed
                perccounter += 1
            xvals.append(2 * i)
            noprio.append(run_model(vperday_fn=1000 * j, vaxcapacityincrease_fn=j * 1000 / 100, vaxmax_fn=j * 1000 * 5,
                                    use_priority_fn=False, vfromday_fn=1 * i)[3].sum(axis=0)[-1])
            oldyoung.append(
                run_model(vperday_fn=1000 * j, vaxcapacityincrease_fn=j * 1000 / 100, vaxmax_fn=j * 1000 * 5,
                          priority_fn=[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                          vfromday_fn=1 * i)[3].sum(axis=0)[-1])
            cmxprio.append(run_model(vperday_fn=1000 * j, vaxcapacityincrease_fn=j * 1000 / 100, vaxmax_fn=j * 1000 * 5,
                                     priority_fn=[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17],
                                     vfromday_fn=1 * i)[3].sum(axis=0)[-1])
            if noprio[-1] > oldyoung[-1] and noprio[-1] > cmxprio[-1]:
                vax_matrix_worst[j - numrunsperdaystarting][i] = 0
            elif oldyoung[-1] > noprio[-1] and oldyoung[-1] > cmxprio[-1]:
                vax_matrix_worst[j - numrunsperdaystarting][i] = 1
            elif cmxprio[-1] > noprio[-1] and cmxprio[-1] > oldyoung[-1]:
                vax_matrix_worst[j - numrunsperdaystarting][i] = 2
            else:
                print("error", noprio[-1], oldyoung[-1], cmxprio[-1])
            if noprio[-1] < oldyoung[-1] and noprio[-1] < cmxprio[-1]:
                vax_matrix[j - numrunsperdaystarting][i] = 0
            elif oldyoung[-1] < noprio[-1] and oldyoung[-1] < cmxprio[-1]:
                vax_matrix[j - numrunsperdaystarting][i] = 1
            elif cmxprio[-1] < noprio[-1] and cmxprio[-1] < oldyoung[-1]:
                vax_matrix[j - numrunsperdaystarting][i] = 2
            else:
                print("error", noprio[-1], oldyoung[-1], cmxprio[-1])
        text_file.write("{},{},{},".format(noprio, oldyoung, cmxprio))
    text_file.close()

# plot heatmap, change formatting, etc.
if format_heatmap:
    with open("heatmap-data-6.txt", "r") as file:
        vaxmxstring = file.read().replace('\n', '')
    vaxmx = ast.literal_eval(vaxmxstring)
    firstones = []
    secondones = []
    thirdones = []
    for i in range(len(vaxmx)):
        for j in vaxmx[i]:
            if i % 3 == 0:
                firstones.append(j)
            elif i % 3 == 1:
                secondones.append(j)
            else:
                thirdones.append(j)
        if i % 3 == 2:
            for j in range(len(firstones)):
                if firstones[j] > secondones[j] and firstones[j] > thirdones[j]:
                    vax_matrix_worst[i // 3][j] = 0
                elif secondones[j] > firstones[j] and secondones[j] > thirdones[j]:
                    vax_matrix_worst[i // 3][j] = 1
                elif thirdones[j] > firstones[j] and thirdones[j] > secondones[j]:
                    vax_matrix_worst[i // 3][j] = 2
                if firstones[j] < secondones[j] and firstones[j] < thirdones[j]:
                    vax_matrix[i // 3][j] = 0
                elif secondones[j] < firstones[j] and secondones[j] < thirdones[j]:
                    vax_matrix[i // 3][j] = 1
                elif thirdones[j] < firstones[j] and thirdones[j] < secondones[j]:
                    vax_matrix[i // 3][j] = 2
            firstones = []
            secondones = []
            thirdones = []


    def get_alpha_blend_cmap(cmap_hm, alpha):
        cls = plt.get_cmap(cmap_hm)(np.linspace(0, 1, 256))
        cls = (1 - alpha) + alpha * cls
        return ListedColormap(cls)


    vax_matrix_columnsx = [1 * i for i in range(numruns)]
    vax_matrix_columns = [format(1000 * j, ",") for j in range(1, numrunsperday + 1)]
    myColors = (palette_five[1], palette_five[2], palette_five[4])
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    fig_width_pt = textwidth
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * (1 / 2) / golden_ratio
    figsize = (fig_width_in, fig_height_in)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, constrained_layout=True, sharey=True)
    vax_matrix_df = pd.DataFrame(data=vax_matrix, columns=vax_matrix_columnsx, index=vax_matrix_columns)
    sns.heatmap(vax_matrix_df, ax=ax1, cmap=get_alpha_blend_cmap(cmap, 0.75), vmin=0, vmax=2, xticklabels=50,
                cbar=False, rasterized=True)
    ax1.set_xlabel("Vaccination delay")
    ax1.set_title("(a) Best strategy.")
    vax_matrix_worst_df = pd.DataFrame(data=vax_matrix_worst, columns=vax_matrix_columnsx, index=vax_matrix_columns)
    sns.heatmap(vax_matrix_worst_df, ax=ax2, cmap=get_alpha_blend_cmap(cmap, 0.75), vmin=0, vmax=2, xticklabels=50,
                cbar=False, rasterized=True)
    ax2.set_xlabel("Vaccination delay")
    ax2.set_title("(b) Worst strategy.")
    yticks = np.linspace(9, len(vax_matrix_columns) - 1, 18, dtype=int)
    yticklabels = [vax_matrix_columns[idx] for idx in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_ylabel("Initial vaccination capacity (IVC)")
    ax1.tick_params(axis="y", direction="out", pad=-2)
    ax1.tick_params(axis="x", direction="out", pad=-2)
    ax2.tick_params(axis="x", direction="out", pad=-2.75)
    plt.draw()
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    cax = fig.add_axes([p0[0], -0.075, p1[2] - p0[0], 0.05])
    fig.colorbar(ax2.get_children()[0], cax=cax, orientation="horizontal")
    colorbar = ax2.collections[0].colorbar
    colorbar.set_ticks([0.3333, 1, 1.6667])
    colorbar.set_ticklabels(["Uniform strategy", "Age-based strategy", "Contact-based strategy"])
    colorbar.ax.tick_params(length=0, labelsize=10.95)
    colorbar.outline.set_linewidth(0)
    ax1.tick_params(axis='both', which='major')
    ax2.tick_params(axis='both', which='major')
    ax1.invert_yaxis()
    plt.savefig('graphics/heatmap-6.pdf', dpi=800, bbox_inches="tight", transparent=True)
    plt.show()

#  vax start date (daily)
if make_vax_start_date_daily:
    grouping_key = ["Under 20", "20--39", "40--59", "60--79", "80+"]
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(352):
            if i % 10 == 0:
                print(i)
            x_data.append(i)
            deaths_datal.append(
                run_model(vfromday_fn=i, use_priority_fn=useprio_key[graphnum], priority_fn=prio_key[graphnum])[3][:,
                -1])
        deaths_data = np.array(deaths_datal)
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday_fn=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([i for i in range(len(grouped_deaths[0]) - 1)], cumulative_to_daily(grouped_deaths[0]),
                               cumulative_to_daily(grouped_deaths[1]), cumulative_to_daily(grouped_deaths[2]),
                               cumulative_to_daily(grouped_deaths[3]), cumulative_to_daily(grouped_deaths[4]),
                               edgecolor="none", alpha=0.9)
        ax[graphnum].set_xlim(right=len(deaths_data[:, 0]) - 1)
        ax[graphnum].set_ylim(top=1000)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="upper right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Days since first infection")
        if graphnum == 1:
            ax[graphnum].set_ylabel("Deaths from single-day additional delay in vaccination")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-startdates-stacked-age-prio3-daily.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax start date (cumulative)
if make_vax_start_date_cum:
    grouping_key = ["Under 20", "20–39", "40–59", "60–79", "80+"]
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(351):
            if i % 10 == 0:
                print(i)
            x_data.append(i)
            deaths_datal.append(
                run_model(vfromday_fn=i, use_priority_fn=useprio_key[graphnum], priority_fn=prio_key[graphnum])[3][:,
                -1])
        deaths_data = np.array(deaths_datal)
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday_fn=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([i for i in range(len(grouped_deaths[0]))], grouped_deaths[0], grouped_deaths[1],
                               grouped_deaths[2], grouped_deaths[3], grouped_deaths[4], edgecolor="none", alpha=0.9)
        ax[graphnum].set_xlim(right=len(deaths_data[:, 0]) - 1)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="center left")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Vaccination delay (in days)")
        ax[graphnum].plot([i for i in range(len(grouped_deaths[0]))], lives_saved, 'k--')
        if graphnum == 1:
            ax[graphnum].set_ylabel("Total number of deaths")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-startdates-stacked-age-prio3-cum.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax start date with 3 diff priorities on same graph
if make_vax_start_date_samegraph:
    plt.figure(figsize=set_size(textwidth))
    sns.set_palette([palette_five[3]])
    x_data = []
    deaths_data = []
    deaths_data2 = []
    deaths_data3 = []
    for i in range(351):
        if i % 10 == 0:
            print(i)
        x_data.append(i)
        model_output = run_model(vfromday_fn=i, use_priority_fn=False)[3].sum(axis=0)
        deaths_data.append(model_output[-1])
        model_output = run_model(vfromday_fn=i, use_priority_fn=True,
                                 priority_fn=[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])[3].sum(
            axis=0)
        deaths_data2.append(model_output[-1])
        model_output = run_model(vfromday_fn=i, use_priority_fn=True,
                                 priority_fn=[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17])[3].sum(
            axis=0)
        deaths_data3.append(model_output[-1])
    xy_dict = {'Vaccination delay (in days)': x_data, 'Total number of deaths': deaths_data}
    df_deaths_data = pd.DataFrame(xy_dict)
    xy_dict_all3 = {'Vaccination delay (in days)': x_data + x_data + x_data,
                    'Total number of deaths': deaths_data + deaths_data2 + deaths_data3,
                    'series': ['Uniform strategy' for i in range(len(x_data))] + ['Age-based strategy' for i in
                                                                                  range(len(x_data))] + [
                                  'Contact-based strategy' for i in range(len(x_data))]}
    df_deaths_data_all3 = pd.DataFrame(data=xy_dict_all3)
    shading(sns.lineplot(x='Vaccination delay (in days)', y='Total number of deaths', hue='series',
                         data=df_deaths_data_all3, dashes=False, linewidth=2), 3, 0.1,
            [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
             (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
             (0.23299120924703914, 0.639586552066035, 0.9260706093977744)], legend=False, xlim=False)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
    ax.set_xlim(right=x_data[-1])
    ax.tick_params(axis="y", direction="out", pad=-3)
    ax.tick_params(axis="x", direction="out", pad=-2)
    plt.savefig('graphics/vax-startdates-samegraph.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

# vax per day (daily)
if make_vax_per_day_daily:
    imult = 5000
    grouping_key = ["Under 20", "20--39", "40--59", "60--79", "80+"]
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(102):
            if i % 10 == 0:
                print(i)
            x_data.append(i * imult)
            deaths_datal.append(
                run_model(vperday_fn=i * imult, vaxcapacityincrease_fn=i * imult / 100, vaxmax_fn=i * imult * 5,
                          use_priority_fn=useprio_key[graphnum], priority_fn=prio_key[graphnum])[3][:, -1])
        deaths_data = np.array(deaths_datal)
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday_fn=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([imult * i for i in range(len(grouped_deaths[0]) - 1)],
                               cumulative_to_daily(grouped_deaths[0], absol=True),
                               cumulative_to_daily(grouped_deaths[1], absol=True),
                               cumulative_to_daily(grouped_deaths[2], absol=True),
                               cumulative_to_daily(grouped_deaths[3], absol=True),
                               cumulative_to_daily(grouped_deaths[4], absol=True), edgecolor="none", alpha=0.9)
        ax[graphnum].set_ylim(top=6000)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="upper right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Initial vaccination capacity (IVC)")
        if graphnum == 1:
            ax[graphnum].set_ylabel("Number of lives saved by increasing initial vaccination capacity by 10,000")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        ax[graphnum].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-capacity-stacked-age-prio3-daily.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax start date (cumulative)
if make_vax_per_day_cum:
    imult = 1000
    grouping_key = ["Under 20", "20–39", "40–59", "60–79", "80+"]
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(501):
            if i % 10 == 0:
                print(i * imult)
            x_data.append(i)
            datatoadd = run_model(vperday_fn=i * imult, vaxcapacityincrease_fn=i * imult / 100, vaxmax_fn=i * imult * 5,
                                  use_priority_fn=useprio_key[graphnum], priority_fn=prio_key[graphnum])
            deaths_datal.append(datatoadd[3][:, -1])
        deaths_data = np.array(deaths_datal)
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday_fn=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([imult * i for i in range(len(grouped_deaths[0]))], grouped_deaths[0], grouped_deaths[1],
                               grouped_deaths[2], grouped_deaths[3], grouped_deaths[4], edgecolor="none", alpha=0.9)
        ax[graphnum].set_xlim(right=imult * len(deaths_data[:, 0]) - 1)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="center right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Initial vaccination capacity (IVC)")
        ax[graphnum].plot([imult * i for i in range(len(grouped_deaths[0]))], lives_saved, 'k--')
        if graphnum == 1:
            ax[graphnum].set_ylabel("Total number of deaths")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        ax[graphnum].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-capacity-stacked-age-prio3-cum.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax per day
if make_vax_per_day_samegraph:
    imult = 1000
    plt.figure(figsize=set_size(textwidth))
    sns.set_palette([palette_five[3]])
    x_data = []
    deaths_data = []
    deaths_data2 = []
    deaths_data3 = []
    for i in range(501):
        if i % 10 == 0:
            print(i, i * imult)
        x_data.append(i * imult)
        model_output = run_model(vperday_fn=i * imult, use_priority_fn=False, vaxcapacityincrease_fn=i * imult / 100,
                                 vaxmax_fn=i * 5 * imult)[3].sum(axis=0)
        deaths_data.append(model_output[-1])
        model_output = run_model(vperday_fn=i * imult, use_priority_fn=True,
                                 priority_fn=[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                                 vaxcapacityincrease_fn=i * imult / 100, vaxmax_fn=i * 5 * imult)[3].sum(axis=0)
        deaths_data2.append(model_output[-1])
        model_output = run_model(vperday_fn=i * imult, use_priority_fn=True,
                                 priority_fn=[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17],
                                 vaxcapacityincrease_fn=i * imult / 100, vaxmax_fn=i * 5 * imult)[3].sum(axis=0)
        deaths_data3.append(model_output[-1])
    xy_dict = {'Initial vaccination capacity (IVC)': x_data, 'Total number of deaths': deaths_data}
    df_deaths_data = pd.DataFrame(xy_dict)
    xy_dict_all3 = {'Initial vaccination capacity (IVC)': x_data + x_data + x_data,
                    'Total number of deaths': deaths_data + deaths_data2 + deaths_data3,
                    'series': ['Uniform strategy' for i in range(len(x_data))] + ['Age-based strategy' for i in
                                                                                  range(len(x_data))] + [
                                  'Contact-based strategy' for i in range(len(x_data))]}
    df_deaths_data_all3 = pd.DataFrame(data=xy_dict_all3)
    shading(sns.lineplot(x='Initial vaccination capacity (IVC)', y='Total number of deaths', hue='series',
                         data=df_deaths_data_all3, dashes=False, linewidth=2), 3, 0.1,
            [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
             (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
             (0.23299120924703914, 0.639586552066035, 0.9260706093977744)], legend=False, xlim=False)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p_lambda: format(int(x), ',')))
    ax.set_xlim(right=x_data[-1])
    ax.tick_params(axis="y", direction="out", pad=-3)
    ax.tick_params(axis="x", direction="out", pad=-2)
    plt.savefig('graphics/vax-capacity.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()
