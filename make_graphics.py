from model import *   
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as mtick
import ast



try_sample_simulation = False
make_vaccine_comparison = False
make_contact_matrix_heatmap = False
make_deaths_by_age_novax = False
make_priority_order = False
make_priority_cum_order = False
make_priority_order_without_novax = False
make_priority_cum_order_without_novax = False
make_heatmap = False
format_heatmap = False
format_heatmap_poster = False
format_heatmap_poster_2 = False
format_heatmap_pres = False
make_deaths_by_priority_ages = False
make_vax_start_date_cum = True
make_vax_start_date_daily = False
make_vax_start_date_samegraph = True
make_vax_per_day = False
make_vax_per_day_daily = True
make_vax_per_day_cum = True
make_vax_per_day_samegraph = True




sns.set(style="whitegrid", rc={'figure.figsize':(10, 6.25)})
palette_five = ['#0f0757','#940061','#e83344','#f06f2b','#ffcc00']
palette_neutral = [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.3126890019504329, 0.6928754610296064, 0.1923704830330379), (0.23299120924703914, 0.639586552066035, 0.9260706093977744)]

textwidth = 418.25368
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10.95,
    "font.size": 10.95,
    'axes.titlesize':10.95,
    # Make the legend/label fonts a little smaller
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

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def run_model(s_date="2020-01-01",e_date="2022-12-15",I0=1,Rval=1.1,priority=[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],use_priority=True,vdelay=22,vprob=0.76,finalvprob=0.813,vdelay2=14,vfromday=100,vperday=100000,dosegap=84,print_model=False,vaxcapacityincrease=1000,vaxmax=500000):
    TotPop = np.dot(1e6, [3.737, 4.136, 4.128, 3.745, 4.057, 4.437, 4.581, 4.434, 4.190, 4.182, 4.604, 4.578, 3.975, 3.395, 3.388, 2.527, 1.733, 1.704])
    Tmax = 700
    asize=Tmax+1
    Kf = [0.000007, 0.000012, 0.000022, 0.000041, 0.000075, 0.000137, 0.000252, 0.000461, 0.000845, 0.001549, 0.002838, 0.0052, 0.009528, 0.017458, 0.031989, 0.058614, 0.107399, 0.283009]
    nP=100
    ndays=int(0.49+(seconds_since_epoch(e_date)-seconds_since_epoch(s_date))/(3600*24))
    ndays=Tmax
    wave = 1
    delay = 14
    contact_matrix = np.loadtxt(open("contact_matrix.csv", "rb"), delimiter=",")*18/400
    pop = PopDyn(TotPop,I0,Tmax,Rval,Kf,nP,wave,contact_matrix,priority,use_priority)
    pop.intialise(wave,delay)
    pop.iterate(ndays, vdelay, vprob, vfromday, vperday, dosegap, finalvprob, vdelay2, vaxcapacityincrease, vaxmax)
    if print_model:
        pop.plot_model(ndays + 1, e_date, ndays + 1, showdates, display_both_lines)
    return(pop.S, pop.I, pop.R, pop.Fat, pop.vaccinated)

def shading(ax,numlines,alpha,palette,legend,xlim,loc=False,commas=True,linewidth=4.0,dashed=False,indivlinestyles=False):
    l=[]
    for i in range(numlines):
        l.append(ax.lines[i])
    x = []
    y = []
    for i in range(numlines):
        x.append(l[i].get_xydata()[:,0])
        y.append(l[i].get_xydata()[:,1])
    for i in range(numlines):
        ax.fill_between(x[i],y[i], color=palette[i], alpha=alpha)
    # sns.despine(left=True,bottom=True)
    ax.set_frame_on(False)
    if xlim != False:
        ax.set_xlim(right=xlim)
    if type(dashed) is list:
        for i in range(len(dashed)):
            if dashed[i]:
                ax.lines[i].set_linestyle("--")
    elif dashed:
        for line in ax.lines:
            line.set_linestyle("--")
    if indivlinestyles:
        for num, line in enumerate(ax.lines[:numlines]):
            print(num,line)
            line.set_linestyle(indivlinestyles[num])
    if legend:
        if loc == False:
            leg = ax.legend()
        else:
            leg = ax.legend(loc=loc)
        for num, line in enumerate(leg.get_lines()):
            line.set_linewidth(linewidth)
            if type(dashed) is list:
                if dashed[num]:
                    line.set_linestyle("--")
            if indivlinestyles:
                line.set_linestyle(indivlinestyles[num])
    if commas:
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

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
        return (nvar1, nvar2)
    else:
        return nvar1

TotPop = np.dot(1e6, [3.737, 4.136, 4.128, 3.745, 4.057, 4.437, 4.581, 4.434, 4.190, 4.182, 4.604, 4.578, 3.975, 3.395, 3.388, 2.527, 1.733, 1.704])
Tmax = 1000
asize=Tmax+1
Kf = [0.000007, 0.000012, 0.000022, 0.000041, 0.000075, 0.000137, 0.000252, 0.000461, 0.000845, 0.001549, 0.002838, 0.0052, 0.009528, 0.017458, 0.031989, 0.058614, 0.107399, 0.283009]
nP=100
s_date="2020-01-01"
e_date="2022-12-15"
ndays=int(0.49+(seconds_since_epoch(e_date)-seconds_since_epoch(s_date))/(3600*24))
ndays=Tmax
wave = 1
delay = 14

I0 = 1
Rval = 1.1

contact_matrix = np.loadtxt(open("contact_matrix.csv", "rb"), delimiter=",")*18/400
priority = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # young -> old
priority = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] # old -> young
use_priority = False

if try_sample_simulation:
    pop = PopDyn(TotPop,I0,Tmax,Rval,Kf,nP,wave,contact_matrix,priority,use_priority)
    pop.intialise(wave,delay)
    print("I0={}, R={}, number of days = {}".format(I0, Rval, ndays+1))

vdelay = 14 # pfizer delay until fully effective after 1 dose
vprob = 0.926 # pfizer effectiveness after 1 dose
finalvprob=0.95 # pfizer effectiveness after 2 doses
vdelay2=7 # pfizer delay until fully effective after 2 doses

vfromday = 100 # how long to wait until vaccinations start
vperday = 30000 # how many vaccines per day
dosegap=84 # gap between doses

vaxcapacityincrease = 0
vaxmax = 150000

W = 4.0
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# plt.rcParams.update({
#     'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
#     'font.size' : 11,                   # Set font size to 11pt
#     'axes.labelsize': 11,               # -> axis labels
#     'legend.fontsize': 11,              # -> legends
#     'font.family': 'lmodern',
#     'text.usetex': True,
#     'text.latex.preamble': (            # LaTeX preamble
#         r'\usepackage{lmodern}'
#         # ... more packages if needed
#     )
# })

if make_vaccine_comparison:
    plt.figure(figsize=set_size(textwidth))
    # sns.set_palette([palette_five[1],palette_five[2],palette_five[4]])
    sns.set_palette(palette_neutral)
    plt.xlabel("Number of days after first dose")
    plt.ylabel("Average level of immunity")
    # plt.title("Linear functions for immunity levels after vaccination")
    nPad=101
    vdelayb = 22 # az delay until fully effective after 1 dose
    vprobb = 0.76 # az efficacy after 1 dose
    finalvprobb=0.813 # az efficacy after 2 doses
    vdelay2b=14 # az delay until fully effective after 2 doses
    dosegapb=84 # gap between doses
    vdelayc = 14 # mod delay until fully effective after 1 dose
    vprobc = 0.921 # mod efficacy after 1 dose
    finalvprobc=0.941 # mod efficacy after 2 doses
    vdelay2c=14 # mod delay until fully effective after 2 doses
    dosegapc=21 # gap between doses
    df_linear = pd.DataFrame(np.column_stack([100-100*np.array([1-(vprob*i/(vdelay-1)) for i in range(vdelay)] + [1-vprob]*(dosegap-vdelay) + [1-vprob-((finalvprob-vprob)*i/(vdelay2-1)) for i in range(vdelay2)] + [1-finalvprob]*(nPad-dosegap-vdelay2)),100-100*np.array([1-(vprobc*i/(vdelayc-1)) for i in range(vdelayc)] + [1-vprobc]*(dosegapc-vdelayc) + [1-vprobc-((finalvprobc-vprobc)*i/(vdelay2c-1)) for i in range(vdelay2c)] + [1-finalvprobc]*(nPad-dosegapc-vdelay2c)),100-100*np.array([1-(vprobb*i/(vdelayb-1)) for i in range(vdelayb)] + [1-vprobb]*(dosegapb-vdelayb) + [1-vprobb-((finalvprobb-vprobb)*i/(vdelay2b-1)) for i in range(vdelay2b)] + [1-finalvprobb]*(nPad-dosegapb-vdelay2b))]),columns=["Pfizer-BioNTech","Moderna","Oxford-AstraZeneca"])
    shading(sns.lineplot(data=df_linear,dashes=False,linewidth=2),3,0.1,palette_neutral,True,100,commas=False,linewidth=2,indivlinestyles=["-","--","-."])
    plt.ylim(-5,100)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis="y",direction="out", pad=-4)
    ax.tick_params(axis="x",direction="out", pad=-2)
    plt.savefig('graphics/vax-comparison.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

vdelay = 22 # az delay until fully effective after 1 dose
vprob = 0.76 # az efficacy after 1 dose
finalvprob=0.813 # az efficacy after 2 doses
vdelay2=14 # az delay until fully effective after 2 doses



if make_contact_matrix_heatmap:
    W = 5.5
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    plt.rcParams.update({
        'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
        'font.size' : 11,                   # Set font size to 11pt
        'axes.labelsize': 11,               # -> axis labels
        'legend.fontsize': 11,              # -> legends
        'font.family': 'lmodern',
        'text.usetex': True,
        'text.latex.preamble': (            # LaTeX preamble
            r'\usepackage{lmodern}'
            # ... more packages if needed
        )
    })
    cmx_columns = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
    cmx_columnsx = ["0-4","5-9"," 10-14"," 15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
    cmx_df = pd.DataFrame(data=contact_matrix, columns = cmx_columnsx, index = cmx_columns)
    with plt.style.context({"xtick.labelsize":11,"ytick.labelsize":11}):
        sns.heatmap(cmx_df, cmap="rocket_r", square=True,vmin=0,vmax=0.7)
    ax = plt.gca()
    # for x in ax.get_xticklabels(which="both"):
    #     x.set_va('top')
    # ax.set_xticklabels(cmx_columns,rotation=70,ha="right")
    plt.xlim(0,18)
    plt.ylim(0,18)
    # plt.xlabel("Age group")
    # plt.ylabel("Age group")
    # plt.title("Contact matrix for five-year age intervals") # for presentation
    # tikzplotlib.save("contact-matrix.tex")
    plt.savefig('graphics/contact-matrix.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

showdates = False
display_both_lines = True

if try_sample_simulation:
    pop.iterate(ndays, vdelay, vprob, vfromday, vperday, dosegap, finalvprob, vdelay2, vaxcapacityincrease, vaxmax)
    pop.plot_model(ndays+1, e_date, ndays+1, showdates, display_both_lines)

    if display_both_lines == True:
        stats = pop.stats()
        vstats = pop.vstats()
        print("Total population:", f'{int(sum(TotPop)):,}')
        print("\nTotal susceptible:", f'{int(round(sum(vstats[0]))):,}')
        print("Total infected:", f'{int(round(sum(vstats[1]))):,}')
        print("Total recovered:", f'{int(round(sum(vstats[2]))):,}')
        print("\nTotal fatalities:", f'{int(round(sum(vstats[3]))):,}')
        print("Total vaccinated:", f'{int(round(sum(vstats[4]))):,}')
        print("\nTotal lives saved by vaccine:", f'{int(round(sum(stats[3])-sum(vstats[3]))):,}')
    else:
        stats = pop.stats()
        print("Total population:", f'{int(sum(TotPop)):,}')
        print("\nTotal susceptible:", f'{int(round(sum(stats[0]))):,}')
        print("Total infected:", f'{int(round(sum(stats[1]))):,}')
        print("Total recovered:", f'{int(round(sum(stats[2]))):,}')
        print("\nTotal fatalities:", f'{int(round(sum(stats[3]))):,}')
        print("Total vaccinated:", f'{int(round(sum(stats[4]))):,}')

if make_deaths_by_age_novax:
    # grouping_key = ["Under 20","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
    grouping_key = ["Under 20","20--39","40--59","60--79","80+"]

    # grouped_deaths = np.zeros((8,len(vstats[5][0])))
    # grouped_deaths[0] = np.sum([vstats[5][0],vstats[5][1],vstats[5][2],vstats[5][3]],axis=0)
    # for i in range(1,8):
    #     grouped_deaths[i] = np.sum([vstats[5][2*i+2],vstats[5][2*i+3]],axis=0)

    grouped_deaths = np.zeros((5,len(vstats[5][0])))
    grouped_deaths[0] = np.sum([vstats[5][0],vstats[5][1],vstats[5][2],vstats[5][3]],axis=0)
    for i in range(1,4):
        grouped_deaths[i] = np.sum([vstats[5][4*i],vstats[5][4*i+1],vstats[5][4*i+2],vstats[5][4*i+3]],axis=0)
    grouped_deaths[4] = np.sum([vstats[5][16],vstats[5][17]],axis=0)

    grouped_deaths_novax = np.zeros((5,len(stats[5][0])))
    grouped_deaths_novax[0] = np.sum([stats[5][0],stats[5][1],stats[5][2],stats[5][3]],axis=0)
    for i in range(1,4):
        grouped_deaths_novax[i] = np.sum([stats[5][4*i],stats[5][4*i+1],stats[5][4*i+2],stats[5][4*i+3]],axis=0)
    grouped_deaths_novax[4] = np.sum([stats[5][16],stats[5][17]],axis=0)

    sns.set_palette(palette_five)

    plt.figure(figsize=(8,4.5)) # for poster
    grouped_deaths = grouped_deaths_novax
    plt.xlabel("Number of days")
    plt.ylabel("Total deaths")
    # plt.title("Cumulative deaths over time for different age groups") # for pres
    df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
    # shading(sns.lineplot(data=df_deaths,dashes=False,linewidth=4),5,0.1,palette_five,True,ndays+1)
    plt.stackplot([i for i in range(len(grouped_deaths[0]))],grouped_deaths[0],grouped_deaths[1],grouped_deaths[2],grouped_deaths[3],grouped_deaths[4], edgecolor="none", alpha=0.9)
    plt.xlim(right=len(grouped_deaths[0])-1)
    plt.legend(grouping_key, loc="upper left")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig('graphics/novax-presentation-ages-cum.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

    # plt.figure(figsize=(8,4.5)) # for poster
    plt.xlabel("Number of days")
    plt.ylabel("Daily deaths")
    # plt.title("Daily deaths over time for different age groups") # for pres
    df_deaths_cum = pd.DataFrame(np.column_stack([pop.cumulative_to_daily(grouped_deaths[i],"")[0] for i in range(5)]), columns=grouping_key)
    # shading(sns.lineplot(data=df_deaths_cum,dashes=False,linewidth=4),5,0.1,palette_five,True,ndays+1)
    plt.stackplot([i for i in range(len(pop.cumulative_to_daily(grouped_deaths[0],"")[0]))],pop.cumulative_to_daily(grouped_deaths[0],"")[0],pop.cumulative_to_daily(grouped_deaths[1],"")[0],pop.cumulative_to_daily(grouped_deaths[2],"")[0],pop.cumulative_to_daily(grouped_deaths[3],"")[0],pop.cumulative_to_daily(grouped_deaths[4],"")[0], edgecolor="none", alpha=0.9)
    plt.xlim(right=len(grouped_deaths[0])-1)
    plt.legend(grouping_key, loc="upper left")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig('graphics/novax-presentation-ages.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()


# priority order
if make_priority_order:
    categories = ["No vaccine","Uniform strategy","Age-based strategy","Contact-based strategy"]
    grouped_deaths = np.column_stack([run_model(vperday=0)[3].sum(axis=0),run_model(vperday=100000,use_priority=False)[3].sum(axis=0),run_model(vperday=100000)[3].sum(axis=0),run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0)])
    plt.title("Cumulative deaths over time")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette(palette_five[::-1])
    plt.xlabel("Number of days")
    plt.ylabel("Total deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,palette_five[::-1],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages-cum.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()
    # for i in range(len(grouped_deaths[-1:][0])):
    #     print(categories[i],"total deaths:",round(grouped_deaths[-1:][0][i]))


# cum priority order
if make_priority_cum_order:
    categories = ["No vaccine","Uniform strategy","Age-based strategy","Contact-based strategy"]
    grouped_deaths = np.column_stack([pop.cumulative_to_daily(run_model(vperday=0)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000,use_priority=False)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0),"")[0]])
    plt.title("Daily deaths over time")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette(palette_five[::-1])
    plt.xlabel("Number of days")
    plt.ylabel("Daily deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,palette_five[::-1],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

# priority order without novax
if make_priority_order_without_novax:
    categories = ["Uniform strategy","Age-based strategy","Contact-based strategy"]
    grouped_deaths = np.column_stack([run_model(vperday=100000,use_priority=False)[3].sum(axis=0),run_model(vperday=100000)[3].sum(axis=0),run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0)])
    # plt.title("Cumulative deaths over time with different vaccine priority groups")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette([palette_five[1],palette_five[2],palette_five[4]])
    plt.xlabel("Number of days")
    plt.ylabel("Total deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,[palette_five[1],palette_five[2],palette_five[4]],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages-without-novax-cum2.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()
    for i in range(len(grouped_deaths[-1:][0])):
        print(categories[i],"total deaths:",round(grouped_deaths[-1:][0][i]))


# cum priority order without novax
if make_priority_cum_order_without_novax:
    categories = ["Uniform strategy","Age-based strategy","Contact-based strategy"]
    grouped_deaths = np.column_stack([pop.cumulative_to_daily(run_model(vperday=100000,use_priority=False)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0),"")[0]])
    plt.title("Daily deaths over time with different vaccine priority groups")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette([palette_five[1],palette_five[2],palette_five[4]])
    plt.xlabel("Number of days")
    plt.ylabel("Daily deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,[palette_five[1],palette_five[2],palette_five[4]],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages-without-novax2.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()


# #################################################### priority order without novax, varying vfromday
# categories = ["Start date","Uniform strategy","Age-based strategy","Contact-based strategy"]
# numruns = 16
# for j in range(6,11):
#     noprio = []
#     oldyoung = []
#     cmxprio = []
#     xvals = []
#     for i in range(numruns):
#         if i%10 == 0:
#             print(i,2*i)
#         xvals.append(20*i)
#         noprio.append(run_model(vperday=10000*j,use_priority=False,vfromday=20*i)[3].sum(axis=0)[-1])
#         oldyoung.append(run_model(vperday=10000*j,vfromday=20*i)[3].sum(axis=0)[-1])
#         cmxprio.append(run_model(vperday=10000*j,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17],vfromday=20*i)[3].sum(axis=0)[-1])
#     print(noprio,oldyoung,cmxprio)
#     grouped_deaths = np.column_stack([xvals,noprio,oldyoung,cmxprio])
#     plt.title("Total deaths under different strategies as vaccination start date varies")
#     df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
#     sns.set_palette([palette_five[3],palette_five[2],palette_five[1]])
#     plt.xlabel("Vaccination start date (days after first case)")
#     plt.ylabel("Total deaths")
#     data=df_deaths_vax
#     print(data)
#     print(pd.melt(data, ["Start date"]))
#     shading(sns.lineplot(x="Start date", y="value", hue="variable", data=pd.melt(data, ["Start date"]),dashes=False,linewidth=4),data.shape[1]-1,0.1,[palette_five[3],palette_five[2],palette_five[1]],True,20*(numruns-1),loc="upper left")
#     plt.savefig('graphics/priorities-ages-without-novax-varying-cum{}.png'.format(10000*j), dpi=400, bbox_inches="tight", transparent=True)
#     plt.show()
# plt.plot(run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17],vfromday=0)[3].sum(axis=0))
# plt.show()
# # for i in range(len(grouped_deaths[-1:][0])):
# #     print(categories[i],"total deaths:",round(grouped_deaths[-1:][0][i]))

# priority order without novax, varying vfromday
categories = ["Start date","Uniform strategy","Age-based strategy","Contact-based strategy"]
numruns = 251
numrunsperday = 350
vax_matrix = np.zeros((numrunsperday,numruns))
vax_matrix_worst = np.zeros((numrunsperday,numruns))
if make_heatmap:
    counter = 0
    percs = [i/10 for i in range(1001)]
    perccounter = 0
    text_file = open("heatmap-data3.txt","w")
    import time
    starttime = time.time()
    maxrate = 999999999999
    oldelapsed = 0
    oldnumruns = 0
    #######
    for j in range(1,numrunsperday+1):
        noprio = []
        oldyoung = []
        cmxprio = []
        xvals = []
        for i in range(numruns):
            if (100*((j-1)*numruns+i)/((numruns)*(numrunsperday))) >= percs[perccounter]:
                elapsed = time.time()-starttime
                hours = round((elapsed/60)//60)
                mins = round((elapsed/60)%60)
                proportiondone = ((j-1)*numruns+i)/((numruns)*(numrunsperday))
                if proportiondone == 0:
                    print("{}% done ({}/{}).".format(round(100*proportiondone,2),3*((j-1)*numruns+i),3*(numruns)*(numrunsperday)))
                else:
                    if elapsed-oldelapsed < maxrate:
                        maxrate = elapsed-oldelapsed
                        print("{}% done ({}/{}). {}h {}m elapsed, {}h {}m left. Running at {}% max speed (new record!), {}s per run.".format(round(100*proportiondone,2),3*((j-1)*numruns+i),3*(numruns)*(numrunsperday),round((elapsed/60)//60),round((elapsed/60)%60),round(((elapsed/proportiondone-elapsed)/60)//60),round(((elapsed/proportiondone-elapsed)/60)%60),round(100*maxrate/(elapsed-oldelapsed),2),round((elapsed-oldelapsed)/((j-1)*numruns+i-oldnumruns),2)))
                    else:
                        print("{}% done ({}/{}). {}h {}m elapsed, {}h {}m left. Running at {}% max speed, {}s per run.".format(round(100*proportiondone,2),3*((j-1)*numruns+i),3*(numruns)*(numrunsperday),round((elapsed/60)//60),round((elapsed/60)%60),round(((elapsed/proportiondone-elapsed)/60)//60),round(((elapsed/proportiondone-elapsed)/60)%60),round(100*maxrate/(elapsed-oldelapsed),2),round((elapsed-oldelapsed)/((j-1)*numruns+i-oldnumruns),2)))
                oldnumruns = (j-1)*numruns+i
                oldelapsed = elapsed
                perccounter += 1
            # counter = (1000*((j-1)*numruns+i)/((numruns)*(numrunsperday)))
            xvals.append(2*i)
            noprio.append(run_model(vperday=1000*j,use_priority=False,vfromday=1*i)[3].sum(axis=0)[-1])
            oldyoung.append(run_model(vperday=1000*j,vfromday=1*i)[3].sum(axis=0)[-1])
            cmxprio.append(run_model(vperday=1000*j,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17],vfromday=1*i)[3].sum(axis=0)[-1])
            if noprio[-1] > oldyoung[-1] and noprio[-1] > cmxprio[-1]:
                vax_matrix_worst[j-1][i] = 0
            elif oldyoung[-1] > noprio[-1] and oldyoung[-1] > cmxprio[-1]:
                vax_matrix_worst[j-1][i] = 1
            elif cmxprio[-1] > noprio[-1] and cmxprio[-1] > oldyoung[-1]:
                vax_matrix_worst[j-1][i] = 2
            else:
                print("error",noprio[-1],oldyoung[-1],cmxprio[-1])
            if noprio[-1] < oldyoung[-1] and noprio[-1] < cmxprio[-1]:
                vax_matrix[j-1][i] = 0
            elif oldyoung[-1] < noprio[-1] and oldyoung[-1] < cmxprio[-1]:
                vax_matrix[j-1][i] = 1
            elif cmxprio[-1] < noprio[-1] and cmxprio[-1] < oldyoung[-1]:
                vax_matrix[j-1][i] = 2
            else:
                print("error",noprio[-1],oldyoung[-1],cmxprio[-1])
        # print(noprio,oldyoung,cmxprio)
        text_file.write("{},{},{},".format(noprio,oldyoung,cmxprio))
    text_file.close()



#######
    # grouped_deaths = np.column_stack([xvals,noprio,oldyoung,cmxprio])
    # plt.title("Total deaths under different strategies as vaccination start date varies")
    # df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    # sns.set_palette([palette_five[3],palette_five[2],palette_five[1]])
    # plt.xlabel("Vaccination start date (days after first case)")
    # plt.ylabel("Total deaths")
    # data=df_deaths_vax
    # print(data)
    # print(pd.melt(data, ["Start date"]))
    # shading(sns.lineplot(x="Start date", y="value", hue="variable", data=pd.melt(data, ["Start date"]),dashes=False,linewidth=4),data.shape[1]-1,0.1,[palette_five[3],palette_five[2],palette_five[1]],True,20*(numruns-1),loc="upper left")
    # plt.savefig('graphics/priorities-ages-without-novax-varying-cum{}.png'.format(10000*j), dpi=400, bbox_inches="tight", transparent=True)
    # plt.show()

######################## FOR FORMATTING
if format_heatmap:
    # vaxmx = [[79654.8112004229, 84766.74071222279, 89579.71926240911, 94076.56569431843, 98248.2582066266, 102090.62266717711, 105601.8702735366, 108779.7694716851, 111617.57118452853, 114096.91989012962, 116175.51022714356, 117772.88439197467, 118787.52008159763, 119215.96561735164, 119288.28994635266, 119289.10654579147],[76556.77932845849, 79185.25502410009, 81864.88621567568, 84670.76225301209, 87749.43808048594, 91264.41215217084, 95126.6189439272, 99029.26411659174, 102921.29245309814, 106780.48927673664, 110545.39469039418, 114058.57581006708, 116976.40698220313, 118784.69569286301, 119280.56265422143, 119289.10654579147],[51067.12105819258, 59184.7055094765, 67308.23541455508, 75228.59869540954, 82779.40659781061, 89832.86875507895, 96296.92477291114, 102105.93001885302, 107206.33492097748, 111537.01919542992, 115004.50980774904, 117469.807323171, 118825.11705973327, 119251.06521627458, 119289.02205532652, 119289.10654579147],[47116.33250262568, 55281.381866507385, 63436.19292385837, 71386.89842395624, 78987.02148169433, 86134.06099780652, 92760.65566943988, 98822.00596132307, 104280.44090660794, 109084.99688282632, 113141.86818000975, 116281.85428110039, 118290.11384667072, 119143.0058496036, 119287.47350038498, 119289.10654579147],[52958.614523149234, 56054.10407562905, 59254.89574362902, 62693.116866239565, 66613.31525267627, 71257.43308099685, 76468.46022994543, 81934.86199164642, 87848.60922842196, 94512.29097005991, 101823.31351877963, 108836.40609200463, 114665.72556547761, 118280.44829307159, 119272.01897492023, 119289.10654579147],[18697.830943000732, 26586.00705499332, 35800.66183001993, 45835.31920842867, 56312.42969622023, 66982.14538295435, 77477.75045229736, 87427.00928899345, 96519.15857730673, 104476.82832326042, 111000.74994429028, 115729.22652736043, 118371.62634222257, 119213.28752378674, 119288.93760749586, 119289.10654579147],[24984.748888421684, 33245.538990531466, 42432.0691844881, 52148.5617656042, 62001.46628999832, 71657.04696289601, 80864.73524598127, 89449.70505005521, 97285.55391396402, 104253.47771054244, 110187.40333522866, 114815.73388260392, 117796.85753549697, 119070.22681153171, 119286.65720786218, 119289.10654579147],[39150.5961708606, 41991.33417167895, 45113.66970685506, 48757.55510774056, 53080.7227797217, 57851.015957499105, 63061.50308406081, 69119.5255014922, 76420.34064698027, 84761.13208194988, 93886.05655899493, 103648.55325122987, 112356.4956290582, 117776.3469165469, 119263.47549655313, 119289.10654579147],[6009.202804414821, 10385.911190723333, 16992.205377441886, 26082.068072102968, 37352.38247446773, 49925.69912184202, 62775.19511811473, 75304.36767210356, 87242.3695889126, 98102.66161243022, 107270.88343075135, 114065.04150637638, 117928.41529224638, 119175.77158984558, 119288.85320227797, 119289.10654579147],[12282.611055211413, 18768.459840501837, 27039.860908904935, 36813.745016544555, 47597.112082025356, 58825.239430133515, 69987.6073263768, 80688.2399842937, 90638.32935114097, 99601.98318514976, 107311.2729795333, 113374.23853735087, 117307.72093504373, 118997.62807284686, 119285.8410687579, 119289.10654579147],[29513.179418781852, 32503.121430259813, 35805.064300297956, 39411.89717702453, 43325.559350874544, 47825.85283025289, 53290.005363245924, 59814.03836081838, 67347.80340135664, 76450.0150314903, 87283.71182381254, 98962.3607495984, 110052.59248315998, 117272.36544512154, 119254.93220627907, 119289.10654579147],[2134.852134650495, 4340.964270585261, 8272.614495269205, 14600.220305384017, 23830.122383493363, 35991.577547065885, 50207.981501278715, 65019.54306841848, 79246.86677098389, 92382.79182504032, 103804.26320084665, 112474.71922627158, 117495.26380155142, 119138.51554126188, 119288.7688396515, 119289.10654579147],[5825.078678863044, 10196.994718067925, 16643.328787666534, 25284.819567707957, 35843.67228718656, 47715.36482379268, 60172.480740438885, 72553.6499742464, 84341.69774486346, 95129.62326175842, 104512.57161065176, 111957.08145206452, 116822.6739266267, 118925.20920398677, 119285.02508304577, 119289.10654579147],[21922.155012538642, 25001.786171271186, 28366.413811421895, 31990.724838321537, 36029.25711137805, 40618.09408412999, 45846.60352677681, 52175.63001068713, 60115.51659182584, 69736.72060701203, 81406.13991511543, 94835.34881392022, 107840.36530179049, 116768.45966903887, 119246.38908941438, 119289.10654579147],[862.3922344286623, 1949.4743889400313, 4176.592796302389, 8392.830601540458, 15524.749316677415, 26065.918629887077, 39810.185998165, 55725.15645374771, 72021.47126812402, 87238.43365161617, 100587.64670900584, 110955.56725353727, 117071.9512746619, 119101.51751032702, 119288.68451959497, 119289.10654579147],[2759.1401232074127, 5466.947965996256, 10052.87892430737, 17044.18609197703, 26590.0960232229, 38312.26854664644, 51430.456081451055, 65052.26012031631, 78396.08810973755, 90835.01037309828, 101790.33395945607, 110563.9738925648, 116341.68648774302, 118852.96977611608, 119284.20925069955, 119289.10654579147],[15887.569594604574, 18907.486147260293, 22198.763698197505, 25881.587694134138, 30006.8014146483, 34599.759435588145, 39927.116261881245, 46237.38949927651, 54028.86704695099, 64034.59861652122, 76403.8107635458, 91045.28471403565, 105799.94058744886, 116264.77619388793, 119237.84612898035, 119289.10654579147],[389.5253623803238, 967.6353562902827, 2277.163037593772, 5014.124398881855, 10226.692544456124, 19045.394798579862, 31827.86170504625, 47776.667378822865, 65343.67481194415, 82467.54468348014, 97600.65414448225, 109504.78754813853, 116658.25711881359, 119064.77563499555, 119288.600242087, 119289.10654579147],[1341.143198595081, 2955.4160030594953, 6050.991809095964, 11383.1001242266, 19527.91202102824, 30520.009550096467, 43741.62300636484, 58180.76824316427, 72799.47219771569, 86716.27734806525, 99143.53816112255, 109194.62537557432, 115864.72869387119, 118780.90936112669, 119283.3935716929, 119289.10654579147],[11225.313721156663, 13919.258503224508, 17125.14395030098, 20759.478456548342, 24830.865456767122, 29510.898863146558, 34870.257948913124, 41229.27750135125, 49061.67746963264, 59077.88529203694, 72033.21070025227, 87618.7405527538, 103906.06755463018, 115764.26580283896, 119229.30330507214, 119289.10654579147],[194.01971488228287, 515.058649265331, 1310.1240754018559, 3138.500761480325, 6959.435293770232, 14059.265719382718, 25532.322839565215, 41174.626678476736, 59362.019677931006, 77974.46955792722, 94793.93236747116, 108119.5865150356, 116253.96120321075, 119028.28805906264, 119288.51600710623, 119289.10654579147],[682.8240079908859, 1639.1314939577899, 3678.690143440109, 7601.363595439459, 14272.888151332529, 24183.06745012486, 37058.78921240015, 51926.78382322843, 67547.46377577854, 82771.09959526442, 96571.1090565318, 107848.74385884925, 115391.77072023661, 118709.02753163766, 119282.57804599953, 119289.10654579147],[7811.186056674032, 10176.224385553145, 12998.148656029554, 16404.482807796623, 20444.846621535722, 25062.061114025717, 30473.172445381435, 36879.85112239836, 44768.03009350297, 54864.270478695165, 68102.60344053475, 84548.17069720564, 102121.70425247896, 115273.69706670719, 119220.76059393983, 119289.10654579147],[107.46979064348386, 295.9928439266654, 793.9686664655135, 2030.970036350855, 4851.715767174175, 10577.399333748659, 20625.19084340162, 35564.605825248815, 54091.926787862874, 73786.9342593256, 92122.14680928674, 106795.56169715355, 115858.84476928892, 118992.05293233633, 119288.43181463116, 119289.10654579147],[369.3610704496076, 944.9674467160457, 2283.907340635848, 5117.296723858469, 10436.26338166403, 19111.62293457915, 31313.136086998886, 46269.76279102413, 62633.46834979905, 78996.72115432634, 94071.92160458725, 106526.0359294041, 114922.78284354416, 118637.32386099568, 119281.76267359308, 119289.10654579147],[5356.577042943725, 7322.932384533631, 9812.56797411198, 12885.376839757864, 16644.647246388024, 21213.478917395456, 26577.61614372115, 33034.77578254428, 40968.90638681475, 51175.02864216149, 64606.631383690124, 81751.80716996474, 100440.40792743542, 114798.76737654215, 119212.21796657812, 119289.10654579147],[65.98088110301134, 184.3644181077733, 509.3338742526091, 1363.788608104243, 3455.0140500555426, 8057.924798178788, 16823.2932558448, 30824.644178918337, 49384.12142602487, 69930.17008678429, 89570.53389267456, 105524.64774571861, 115472.70193994469, 118956.06841080377, 119288.34766464045, 119289.10654579147],[213.84217842949084, 571.2269078497916, 1460.2160396019628, 3497.1109452256724, 7670.821182786234, 15105.189483043556, 26420.865897792944, 41182.24875370565, 58048.87659796443, 75389.98415105925, 91644.80438577088, 105226.20698960841, 114457.73544367618, 118565.79792327492, 119280.9474544473, 119289.10654579147],[3584.4422788210723, 5222.204594821686, 7338.177732636875, 10065.200420512592, 13520.575211387519, 17827.082500857654, 23147.416241864965, 29573.633419884, 37579.219711883015, 47858.69602149253, 61499.036011345604, 79168.99753426612, 98864.53949545992, 114340.3535243767, 119203.6753866048, 119289.10654579147],[44.157753963103204, 123.84724034674767, 346.65720652251093, 953.4870983979408, 2521.4030798596414, 6215.4015520527755, 13808.288227230803, 26844.64733445739, 45159.11682809451, 66373.03377156174, 87141.97488844668, 104298.68780887319, 115095.3481855161, 118920.33265679138, 119288.26355711263, 119289.10654579147],[132.5662360955084, 363.56027782063677, 966.7608234247228, 2439.1301422021224, 5691.422221570096, 11970.89309631247, 22289.901476238596, 36631.310916900125, 53783.29308905166, 71947.36113910879, 89288.54317708538, 103948.96144093816, 113996.59900535742, 118494.4492932774, 119280.13238853587, 119289.10654579147],[2311.2727982539154, 3653.1390378545207, 5450.2604385110035, 7824.669888854085, 10941.80751410256, 14964.56865718014, 20076.520505266497, 26479.226064110364, 34491.94491120005, 44861.8085278401, 58690.45067577449, 76786.74389870446, 97390.99097072524, 113898.65781555038, 119195.13281957008, 119289.10654579147],[31.613367675451144, 88.62412322529046, 249.3060652296156, 695.2109072671367, 1890.0466302834336, 4864.796078099855, 11402.739513440403, 23450.280026057124, 41387.782572062395, 63066.02807519782, 84839.71705714976, 103112.51943277764, 114726.52697026424, 118884.84383919853, 119288.17949202642, 119289.10654579147],[87.56277581394264, 243.58406446551388, 664.5928990446024, 1742.7587462006895, 4277.263794879734, 9535.131675677, 18825.881813949134, 32580.063343738933, 49824.79074660923, 68664.98977214306, 87001.88457854965, 102694.00286517694, 113539.34411978537, 118423.27754653303, 119279.31747583247, 119289.10654579147],[1430.6443097211816, 2485.109731673587, 3995.9516245997393, 6057.189235332687, 8832.552757878242, 12541.97022094318, 17395.411015148966, 23675.664162209534, 31680.827162539408, 42127.49357443264, 56115.49012505225, 74594.46441890854, 96006.69312614494, 113470.92067576884, 119186.59042601308, 119289.10654579147],[23.844482983952467, 66.70723266277784, 187.92446750062953, 527.4883335304439, 1457.6225304600675, 3870.014373620873, 9487.09087973696, 20533.101360937704, 37998.22705781885, 59982.13448702849, 82654.87788830957, 101963.31344287551, 114365.82570425949, 118849.60014305271, 119288.09546936039, 119289.10654579147],[61.140002221268176, 171.21502322418684, 474.43038508363077, 1278.2465008065737, 3264.2862333500198, 7649.203202287294, 15936.979423135555, 28989.15835303146, 46160.18102935737, 65538.70922739037, 84783.53967161317, 101461.03420287235, 113085.94148622805, 118352.28225929971, 119278.50271631082, 119289.10654579147],[858.386139171868, 1644.929761434552, 2875.324084808356, 4653.4116136234325, 7118.861168524404, 10495.158939019275, 15074.097095640573, 21143.699727604966, 29116.719120873593, 39601.43379714265, 53740.74175918898, 72570.45673201059, 94699.01974455197, 113057.93844671336, 119178.04952966112, 119289.10654579147],[18.73377027839279, 52.26529307300881, 147.23829447886584, 414.50237546478655, 1155.990663662493, 3133.7317134403106, 7961.43278070153, 18031.978333087758, 34921.628609446256, 57108.43615883087, 80574.89619894601, 100850.58572426347, 114012.713869523, 118814.59987615873, 119288.0114890931, 119289.10654579147],[44.7433473069547, 125.59448980760598, 351.0261375457212, 962.9440710877757, 2534.034728815083, 6190.423745014534, 13537.355989550604, 25818.165605624563, 42775.289920924995, 62564.097793785186, 82632.18769022907, 100249.75792885684, 112636.36191358668, 118281.46300856351, 119277.68810994463, 119289.10654579147],[510.60221902912025, 1066.4774381683267, 2032.3927353050767, 3536.361892329588, 5716.379160673166, 8784.780439018317, 13052.759274020507, 18893.52918940962, 26756.62601869112, 37264.24678113485, 51542.972348617375, 70691.37425665534, 93461.19615870767, 112657.64247436305, 119169.51264559275, 119289.10654579147],[15.20465867940936, 42.28642134439399, 119.05399074978214, 335.5699459298443, 940.6198845395526, 2584.5755988687215, 6746.105221546959, 15894.375194701368, 32119.97520690709, 54417.21244947249, 78587.77586558007, 99773.66589082393, 113666.66026037573, 118779.84180272407, 119287.92755120323, 119289.10654579147],[34.04947177679233, 95.58849261491137, 268.33468477006943, 744.535007781438, 2002.6052754483535, 5060.414089391829, 11549.306146957097, 23026.773336732855, 39655.230494419964, 59736.51104547446, 80546.47968586157, 99059.87622466036, 112190.57632192587, 118210.8193720388, 119276.8736567076, 119289.10654579147],[306.55574932600956, 687.9180423573664, 1416.9482045465834, 2657.701795292318, 4568.105335510204, 7345.789970525831, 11309.264893303047, 16891.710080216068, 24589.324032058794, 35099.46925509144, 49493.158953408536, 68932.84688172716, 92286.61091693751, 112271.53996013376, 119160.9874432921, 119289.10654579147],[12.668629588375534, 35.11377004387988, 98.77207799090273, 278.5254070246181, 782.8972447824355, 2169.8587180535083, 5776.899530880704, 14070.292466391202, 29573.856701329572, 51878.47868699046, 76686.84659443001, 98731.54783051534, 113327.22615149747, 118745.32535186427, 119287.84365566942, 119289.10654579147],[26.76619812170601, 75.06562266206785, 211.13729445114728, 589.8846855448778, 1611.2689615161546, 4181.953672522941, 9904.28680108481, 20575.7747237514, 36784.663933559386, 57051.12004295887, 78525.04216839437, 97891.09114764452, 111748.5557439695, 118140.35092816825, 119276.05935657343, 119289.10654579147],[188.33626503884466, 446.8557795361209, 984.0219266792183, 1980.1432436247576, 3627.6616291100177, 6134.927482066806, 9807.076107536132, 15107.50926104902, 22616.403495724604, 33084.62809129628, 47571.51451753686, 67280.90705212357, 91171.97609211609, 111897.70286387639, 119152.48441491943, 119289.10654579147],[10.784853698705662, 29.78515174274061, 83.6956710923246, 236.02707700311214, 664.4767485005718, 1851.7877830883597, 5001.074339560387, 12515.6370477834, 27265.87070716168, 49470.428921149134, 74863.61583782137, 97721.00287675965, 112994.01554337237, 118711.05164464118, 119287.75980247025, 119289.10654579147],[21.618066132113142, 60.52980062827773, 170.3630971389168, 477.88298327522136, 1319.172304418793, 3495.3771846209856, 8543.094279789744, 18427.826778771243, 34148.04232071701, 54502.94903809253, 76566.48070572819, 96743.10479670357, 111310.27132656352, 118070.05725612302, 119275.24520951587, 119289.10654579147],[119.27880207696035, 295.3310537328682, 686.3233320173093, 1470.0714314218008, 2867.4671617769754, 5112.522232221585, 8507.473115765742, 13526.434877116157, 20821.294924168917, 31200.904080958746, 45763.4533314967, 65720.46897169997, 90110.1369673471, 111533.87638190301, 119144.00207114307, 119289.10654579147],[9.34695219744561, 25.717660651755757, 72.18363911518831, 203.53617152429996, 573.5262416876956, 1603.9752716202775, 4376.370827621424, 11193.046095463866, 25179.563122505235, 47185.58924149681, 73110.85494501643, 96740.84392822582, 112666.86697758496, 118677.0202620466, 119287.67599158437, 119289.10654579147],[17.861845797368844, 49.91397097937634, 140.47781935734992, 394.9438481341247, 1097.9440455965525, 2955.096829522584, 7415.454170475901, 16547.98894572432, 31729.828103167372, 52086.91220363888, 74669.38346581049, 95615.61947438537, 110875.69433210473, 117999.93793580291, 119274.43121550858, 119289.10654579147],[78.13897953837012, 199.69062722558013, 484.4170485237103, 1093.737715543049, 2259.9473838945287, 4251.439061272978, 7382.863196607975, 12126.863193865534, 19185.056916028116, 29446.010298611258, 44064.48394261025, 64250.16800780631, 89103.9698703138, 111184.7317799836, 119135.5659071121, 119289.10654579147],[8.223412184112057, 22.539271171930316, 63.18602644913525, 178.12315534153203, 502.1893435285898, 1407.7505711052181, 3869.1531636684417, 10067.95697120889, 23296.83391731197, 45019.11164642641, 71415.50575834315, 95787.20646713208, 112345.50010648322, 118643.23519137109, 119287.59222299047, 119289.10654579147],[15.045708971857088, 41.951353176423, 118.01769358278777, 332.20565296988093, 927.8423672718536, 2526.539852192579, 6479.254052901195, 14904.06264644824, 29514.686788835912, 49797.84896085341, 72832.32468588972, 94508.33784529564, 110444.79613993669, 117929.99254783627, 119273.61737452532, 119289.10654579147],[52.98980482958013, 138.5614876020934, 347.53062064880174, 819.5634388708345, 1782.3819730417204, 3531.186580762584, 6406.968505783889, 10884.408645453908, 17695.44127318428, 27814.873763407682, 42461.18346962491, 62854.69518536228, 88144.52123096536, 110844.52448618074, 119127.16915644077, 119289.10654579147],[7.327726738192361, 20.005505039487744, 56.01217291544927, 157.85154173689483, 445.18366433985295, 1249.9478980890917, 3453.5568605339827, 9109.517482798987, 21601.584668538086, 42969.65994799836, 69770.01155776788, 94858.91666979493, 112029.83510866425, 118609.69564900159, 119287.50849666714, 119289.10654579147],[12.884355543978511, 35.83868669630074, 100.75683810410784, 283.7960900082005, 795.0619965008349, 2183.601421161057, 5699.598843117593, 13466.761378464993, 27487.65098777132, 47630.55753735326, 71053.86805493207, 93420.96309066207, 110017.54824771214, 117860.22067358019, 119272.80368653983, 119289.10654579147],[37.0972404038144, 98.78048605676393, 254.14146053548035, 620.8875522566761, 1410.9416832031181, 2932.7991116718767, 5560.623389565063, 9781.161185228555, 16343.388433514532, 26300.38987580327, 40945.582154828844, 61529.4566456109, 87231.89512466238, 110517.13190563285, 119118.84173946793, 119289.10654579147],[6.601531873224534, 17.951292456441212, 50.1954148186504, 141.40925352391582, 398.89152032161843, 1121.2441643778327, 3109.901880261841, 8291.020285472976, 20078.736262926854, 41035.13963939382, 68169.15806150541, 93954.20025869254, 111719.72080586475, 118576.40154365794, 119287.42481259306, 119289.10654579147],[11.19148435193633, 31.050331410067443, 87.22686987567849, 245.75518977660184, 689.8777944025271, 1906.6057520755, 5047.8167415333555, 12209.74440658368, 25634.255317640524, 45579.826449975255, 69332.56999635394, 92353.19905894094, 109593.92227272227, 117790.62189512062, 119271.9901515258, 119289.10654579147],[26.73976209219765, 72.27672182270743, 189.67783851559346, 476.6911399771804, 1124.2598272673852, 2440.3057573920914, 4828.729751768625, 8799.622908895251, 15114.746280946054, 24891.125327424015, 39508.81950974854, 60264.39129404397, 86355.11685873219, 110196.6182641423, 119110.5189612024, 119289.10654579147],[6.003370996159969, 16.259378996068474, 45.404128519768136, 127.86227407831143, 360.718946631796, 1014.790716501861, 2822.7539703676257, 7588.825555532344, 18711.524650646676, 39210.50535718395, 66608.82376196118, 93070.7475439023, 111414.97379770153, 118543.35132136327, 119287.3411707469, 119289.10654579147],[9.84178328839672, 27.23234856158575, 76.43439452964779, 215.3630494415747, 605.3751900617306, 1680.717010272836, 4500.498828228117, 11109.546778496351, 23940.642872686185, 43640.463671212245, 67666.98283949, 91304.75041236402, 109173.88995319276, 117721.19579527232, 119271.17676945696, 119289.10654579147],[19.81462254855949, 54.16142614017212, 144.52020857556658, 371.4352022219335, 903.6052785745467, 2037.5871515876975, 4197.587393006153, 7924.9277125035305, 13996.628614875217, 23579.496035487067, 38147.58411624887, 59057.2774937635, 85517.62108381704, 109890.49090816209, 119102.36877284964, 119289.10654579147],[5.504416932614866, 14.848220899413983, 41.40763438020945, 116.56027700045162, 328.85245009907374, 925.7275961042832, 2580.68399047011, 6984.14255727069, 17485.617176905376, 37493.65619652375, 65089.27640436424, 92206.38221362396, 111115.38976806944, 118510.541511721, 119287.2575711074, 119289.10654579147],[8.748728638401497, 24.1402175332833, 67.69126884479559, 190.71640460655576, 536.5850798745431, 1494.7238859187391, 4038.620246433818, 10145.435568950681, 22393.644886305825, 41807.32330109365, 66055.65786952358, 90275.32276933065, 108757.42314954704, 117651.94195757897, 119270.36354030702, 119289.10654579147],[15.082577898090939, 41.497512175692506, 112.28428439909915, 293.9158816478198, 733.5612459937976, 1709.8524994081827, 3655.9021718900394, 7146.143148357958, 12979.118709745366, 22360.897308184278, 36859.46795968229, 57902.86424064698, 84711.4867551499, 109589.79926797122, 119094.21816510303, 119289.10654579147],[5.083103512824316, 13.656676233123392, 38.03286980812715, 107.01497491368178, 301.92599639073615, 850.3475794601458, 2374.6217564181065, 6460.535593546907, 16385.36239185451, 35879.369945108476, 63608.719966666526, 91357.34992360031, 110820.61653069798, 118477.98688804056, 119287.17401365377, 119289.10654579147],[7.851175599713157, 21.601082421047998, 60.510273632010794, 170.45881494791067, 479.89443525466066, 1340.1222083509385, 3646.7657124658886, 9299.21802102499, 20980.835893105905, 40075.32962725487, 64497.14824693135, 89264.6228425618, 108344.49384563672, 117582.85996631326, 119269.55046404974, 119289.10654579147],[11.779787356119657, 32.478481911227924, 88.83615591503542, 236.20014760944807, 602.0314363259042, 1443.805467048548, 3192.874059825836, 6453.089829234156, 12051.682557594318, 21228.394383881794, 35640.36768971097, 56798.02070361595, 83935.5826697557, 109295.62709333567, 119086.07451643169, 119289.10654579147],[4.723852245819247, 12.640819825486478, 35.155586863602934, 98.87570223431597, 278.95686706651264, 785.964633436583, 2197.84300852033, 6005.135691795254, 15398.004806484865, 34365.60681210028, 62169.60385983067, 90522.68035807599, 110530.53902042555, 118445.66996135068, 119287.09049837236, 119289.10654579147],[7.105010404337742, 19.490206390919376, 54.53948878233918, 153.60636929032006, 432.64369786624064, 1210.4287678578753, 3312.4653471315505, 8555.022474071888, 19690.567131756645, 38439.498512934755, 62990.01178882063, 88272.3585729422, 107935.07414993938, 117513.94940647682, 119268.7375406589, 119289.10654579147],[9.420437745076025, 25.945034444117752, 71.4839101311534, 192.62304912872858, 499.64800672738505, 1227.7178615577857, 2798.150216761113, 5837.023819772549, 11205.490324489809, 20175.660169150542, 34486.64289414612, 55740.838376302076, 83192.33093133676, 109015.29163026811, 119078.18690296035, 119289.10654579147],[4.414428533578761, 11.766076574037971, 32.6779376460252, 91.86615712325809, 259.16947087973466, 730.4443472095384, 2044.8718342854438, 5606.6681855918605, 14510.279565766166, 32947.03200272882, 60770.97371256782, 89700.06204413025, 110244.90961002259, 118413.57103651654, 119287.00702524315, 119289.10654579147],[6.477810527318418, 17.715896134814148, 49.52006132806815, 139.43338085445325, 392.8504145377888, 1100.6710567337614, 3025.6366134719106, 7899.068254877825, 18511.981153409255, 36894.956104592115, 61532.813605857926, 87298.23925898821, 107529.13629672359, 117445.2098638004, 119267.92477010815, 119289.10654579147],[7.696131009069425, 21.13369994867897, 58.46023954252484, 159.26548717428832, 419.3851251491449, 1051.9085924191027, 2462.3922570030936, 5290.368363816581, 10432.90117871781, 19196.20007081253, 33393.021166705636, 54724.78926839601, 82473.12984105854, 108739.56892385443, 119070.2992232656, 119289.10654579147],[4.14561037669045, 11.006129900795662, 30.525332665900283, 85.77560773727609, 241.97166110854903, 682.1502206772384, 1911.443902297617, 5255.948163888481, 13710.360359425496, 31617.987288547098, 59411.40766017078, 88886.76893191607, 109963.3958181378, 118381.74412458809, 119286.92359435697, 119289.10654579147],[5.945344612630539, 16.209614950968437, 45.25842623024511, 127.39621759878892, 359.01823820502005, 1007.0090544836571, 2778.122611178996, 7319.436436962799, 17435.01064764162, 35436.95489690035, 60124.12858978045, 86341.9756818876, 107126.65264718179, 117376.64092474376, 119267.1121523713, 119289.10654579147],[6.408602067692561, 17.52641221319545, 48.5571426276799, 133.39418849782484, 355.88414176840774, 908.3620334466676, 2176.9671387561, 4806.098039452392, 9727.4062092921, 18284.45455833898, 32357.098019220673, 53749.56234972406, 81779.00072172847, 108471.52620906639, 119062.41383953845, 119289.10654579147],[3.9105569097629953, 10.341869430742726, 28.643775631155812, 80.45155077105312, 226.93473544213026, 639.8950980193326, 1794.4406985408907, 4946.118046959251, 12989.346559857153, 30376.123815677878, 58094.04052553686, 88083.69883534856, 109685.90163331007, 118350.12054145715, 119286.84020574801, 119289.10654579147],[5.489227124740871, 14.91935537065195, 41.607652765285835, 117.08170918665456, 330.0038281487539, 926.4551989885751, 2563.314573794458, 6805.849732866991, 16450.36440743751, 34060.887232565125, 58762.54374773432, 85403.28022606732, 106727.59569053102, 117308.24217649584, 119266.29968742204, 119289.10654579147],[5.4284816545154415, 14.773401467928547, 40.93211526539956, 113.10184556250795, 305.14888324457985, 790.6605100340828, 1934.26058792003, 4377.719101383651, 9083.313376350663, 17435.252593606783, 31375.325833138722, 52811.90857872804, 81106.72932124129, 108207.36064610883, 119054.5377956286, 119289.10654579147],[3.7035286065923616, 9.756849185440903, 26.986616011958038, 75.76211285477395, 213.68755833184505, 602.6475652750652, 1691.113785492398, 4670.826587989954, 12337.526440184016, 29215.388501450758, 56817.70869879486, 87289.39751785772, 109412.04484865318, 118318.75503172542, 119286.75685974408, 119289.10654579147],[5.0953150232409925, 13.805107431493592, 38.4546755231366, 108.17162855695356, 304.9232176236521, 856.6669179184953, 2375.845967387645, 6349.46683355828, 15549.503164133428, 32762.296347471703, 57446.66038089515, 84481.86699525561, 106331.93804508125, 117240.01320697457, 119265.48737523417, 119289.10654579147],[4.669051883036954, 12.636575260565477, 34.98135458202165, 97.02054984908311, 264.2081726824393, 693.61783635308, 1727.5300461749434, 3998.97918737719, 8495.149892285383, 16643.439563891574, 30444.667530888833, 51911.2484296449, 80460.0725926183, 107956.3603952764, 119047.01416665287, 119289.10654579147],[3.519838502055097, 9.23799008965472, 25.516896630374678, 71.60286096300294, 201.9360745069586, 569.5888695820865, 1599.2644738604615, 4424.860838284278, 11746.360308068342, 28129.81232317744, 55580.86819216049, 86503.01655953305, 109141.54511796626, 118287.5823713796, 119286.67355635494, 119289.10654579147],[4.752591562379855, 12.835703176230023, 35.711396737301186, 100.41785004638919, 283.08526507417236, 795.7925882418251, 2211.3462660492814, 5942.694220706327, 14724.607766592664, 31536.885098148523, 56175.09610597135, 83577.4519240134, 105939.65245927231, 117171.95360482717, 119264.67521578136, 119289.10654579147],[4.071189626700591, 10.952324990485378, 30.27361140616805, 84.15490508100267, 230.86859917864092, 613.1054244742611, 1551.0415927951192, 3664.2158580136356, 7958.166877884847, 15904.556625250836, 29562.77777160894, 51046.246147855636, 79833.78259007641, 107710.94883237206, 119039.49294709988, 119289.10654579147],[3.356190915600502, 8.775745699302272, 24.207493015009224, 67.89708465869512, 191.46424044644908, 540.1171699820436, 1517.2726634798548, 4204.350054184147, 11209.499885440146, 27116.04540390474, 54385.36136336575, 85725.54242337376, 108874.15617231358, 118256.68822354206, 119286.59029736751, 119289.10654579147]]
    vaxmx = [[79654.8112004229, 82246.44546257757, 84766.74071222279, 87211.9269741384, 89579.71926240911, 91868.36481146366, 94076.56569431843, 96203.40238312952, 98248.2582066266, 100210.7441879114, 102090.62266717711, 103887.72669779311, 105601.8702735366, 107232.7417817656, 108779.7694716851, 110241.94313571927, 111617.57118452853, 112903.94907546377, 114096.91989012962, 115190.33572238457, 116175.51022714356, 117040.93799007838, 117772.88439197467, 118357.8155398267, 118787.52008159763, 119066.22107960461],[76556.77932845849, 77867.213993172, 79185.25502410009, 80515.42819392278, 81864.88621567568, 83244.65856163266, 84670.76225301209, 86164.3449897182, 87749.43808048594, 89447.02638305575, 91264.41215217084, 93176.91909094763, 95126.6189439272, 97078.63675956466, 99029.26411659174, 100977.46095054285, 102921.29245309814, 104857.46423648173, 106780.48927673664, 108681.34090016829, 110545.39469039418, 112349.52430315843, 114058.57581006708, 115622.41075570771, 116976.40698220313, 118049.74863202915],[51067.12105819258, 55108.94801001943, 59184.7055094765, 63260.034217894456, 67308.23541455508, 71305.09734628143, 75228.59869540954, 79059.09182337168, 82779.40659781061, 86374.81422049936, 89832.86875507895, 93143.14991585078, 96296.92477291114, 99286.73785340869, 102105.93001885302, 104748.07737385908, 107206.33492097748, 109472.67254496632, 111537.01919542992, 113386.41876463912, 115004.50980774904, 116372.05008519275, 117469.807323171, 118285.49013034784, 118825.11705973327, 119124.58589624114],[47116.33250262568, 51185.931873126945, 55281.381866507385, 59373.3090864581, 63436.19292385837, 67447.31075739331, 71386.89842395624, 75238.14027900803, 78987.02148169433, 82622.07613499061, 86134.06099780652, 89515.57750500835, 92760.65566943988, 95864.30259397994, 98822.00596132307, 101629.16917629476, 104280.44090660794, 106768.89177021009, 109084.99688282632, 111215.43434061867, 113141.86818000975, 114840.25212986434, 116281.85428110039, 117437.96894755338, 118290.11384667072, 118844.44765290924],[52958.614523149234, 54497.82309759486, 56054.10407562905, 57635.54363885005, 59254.89574362902, 60931.57532945598, 62693.116866239565, 64574.52044580261, 66613.31525267627, 68839.18536759069, 71257.43308099685, 73825.56811078174, 76468.46022994543, 79163.9597628212, 81934.86199164642, 84815.80118274747, 87848.60922842196, 91074.03932968667, 94512.29097005991, 98134.48740869213, 101823.31351877963, 105423.86416500152, 108836.40609200463, 111960.16649004562, 114665.72556547761, 116811.10522078132],[18697.830943000732, 22446.1046048594, 26586.00705499332, 31059.08827011334, 35800.66183001993, 40744.821462865395, 45835.31920842867, 51033.14515233223, 56312.42969622023, 61643.64012997711, 66982.14538295435, 72276.48012125822, 77477.75045229736, 82541.38238173531, 87427.00928899345, 92097.78124259555, 96519.15857730673, 100657.25353874217, 104476.82832326042, 107939.20805893, 111000.74994429028, 113613.29398353976, 115729.22652736043, 117314.61643117742, 118371.62634222257, 118962.33982198453],[24984.748888421684, 28976.42170881623, 33245.538990531466, 37747.02130899663, 42432.0691844881, 47249.543773857185, 52148.5617656042, 57080.672422498, 62001.46628999832, 66871.56794620669, 71657.04696289601, 76329.332336473, 80864.73524598127, 85243.67839388613, 89449.70505005521, 93468.30487193853, 97285.55391396402, 100886.53264603356, 104253.47771054244, 107363.68472809679, 110187.40333522866, 112686.50828640118, 114815.73388260392, 116529.45638584951, 117796.85753549697, 118623.781111958],[39150.5961708606, 40547.5159613644, 41991.33417167895, 43503.75435818072, 45113.66970685506, 46854.6044054887, 48757.55510774056, 50838.799699749696, 53080.7227797217, 55427.137235743474, 57851.015957499105, 60380.069295689405, 63061.50308406081, 65954.09533299741, 69119.5255014922, 72605.78859726843, 76420.34064698027, 80498.5964669826, 84761.13208194988, 89215.73409825638, 93886.05655899493, 98743.32508405347, 103648.55325122987, 108300.77288121673, 112356.4956290582, 115573.05069684522],[6009.202804414821, 7949.118752605652, 10385.911190723333, 13384.186448117336, 16992.205377441886, 21230.36130037829, 26082.068072102968, 31488.86662746367, 37352.38247446773, 43544.54835962964, 49925.69912184202, 56367.631310292156, 62775.19511811473, 69095.61189952384, 75304.36767210356, 81371.90057244635, 87242.3695889126, 92844.22053925494, 98102.66161243022, 102939.45847380193, 107270.88343075135, 111008.26600830452, 114065.04150637638, 116375.61462899526, 117928.41529224638, 118802.33702439349],[12282.611055211413, 15295.268302117664, 18768.459840501837, 22692.42914215173, 27039.860908904935, 31766.357659967034, 36813.745016544555, 42114.584855117515, 47597.112082025356, 53189.83607069174, 58825.239430133515, 64442.274200885935, 69987.6073263768, 75415.74396567476, 80688.2399842937, 85772.21568612153, 90638.32935114097, 95258.29563654022, 99601.98318514976, 103634.16698361616, 107311.2729795333, 110579.14967397157, 113374.23853735087, 115632.16751865446, 117307.72093504373, 118404.2163176874],[29513.179418781852, 30975.8277277264, 32503.121430259813, 34110.141455787176, 35805.064300297956, 37577.987167395564, 39411.89717702453, 41317.49367625027, 43325.559350874544, 45478.25764080951, 47825.85283025289, 50418.16250089494, 53290.005363245924, 56437.06577083745, 59814.03836081838, 63425.421626049625, 67347.80340135664, 71667.85537442303, 76450.0150314903, 81693.39962722741, 87283.71182381254, 93073.38766513429, 98962.3607495984, 104754.5614660459, 110052.59248315998, 114335.34143670816],[2134.852134650495, 3065.2806233651327, 4340.964270585261, 6047.317101106871, 8272.614495269205, 11099.590656730916, 14600.220305384017, 18832.526151401533, 23830.122383493363, 29577.4472453864, 35991.577547065885, 42929.602497783504, 50207.981501278715, 57630.10450715574, 65019.54306841848, 72248.63563141302, 79246.86677098389, 85976.12942002088, 92382.79182504032, 98368.53187737586, 103804.26320084665, 108551.35885422403, 112474.71922627158, 115467.5881770771, 117495.26380155142, 118644.54629247056],[5825.078678863044, 7772.6006591015575, 10196.994718067925, 13144.513310280021, 16643.328787666534, 20697.58745079334, 25284.819567707957, 30356.78903563071, 35843.67228718656, 41660.63253243848, 47715.36482379268, 53915.16928335065, 60172.480740438885, 66408.3234834377, 72553.6499742464, 78548.83369727468, 84341.69774486346, 89884.42659206179, 95129.62326175842, 100025.76535097946, 104512.57161065176, 108517.60429798372, 111957.08145206452, 114745.99208329448, 116822.6739266267, 118185.748145767],[21922.155012538642, 23420.42750890896, 25001.786171271186, 26656.140622694278, 28366.413811421895, 30138.450511791925, 31990.724838321537, 33946.431251142036, 36029.25711137805, 38254.22332021049, 40618.09408412999, 43131.29458404386, 45846.60352677681, 48836.01243519532, 52175.63001068713, 55928.57748696349, 60115.51659182584, 64708.9131379949, 69736.72060701203, 75287.48730350293, 81406.13991511543, 88011.17603027596, 94835.34881392022, 101564.72937000603, 107840.36530179049, 113100.73885379882],[862.3922344286623, 1304.497699598839, 1949.4743889400313, 2873.9850274127143, 4176.592796302389, 5974.567955060569, 8392.830601540458, 11547.336915430446, 15524.749316677415, 20366.314154661264, 26065.918629887077, 32577.79457492948, 39810.185998165, 47601.317239033575, 55725.15645374771, 63937.34065970838, 72021.47126812402, 79820.66582248194, 87238.43365161617, 94198.53111933041, 100587.64670900584, 106236.19398955001, 110955.56725353727, 114589.62156945468, 117071.9512746619, 118488.93652086027],[2759.1401232074127, 3918.0619015442753, 5466.947965996256, 7486.593367380608, 10052.87892430737, 13226.694509494666, 17044.18609197703, 21509.21306326771, 26590.0960232229, 32221.687987325036, 38312.26854664644, 44753.41972279044, 51430.456081451055, 58231.266095899846, 65052.26012031631, 71801.042421817, 78396.08810973755, 84764.00702613982, 90835.01037309828, 96537.17565302021, 101790.33395945607, 106501.2858657953, 110563.9738925648, 113870.81991641683, 116341.68648774302, 117968.37148664854],[15887.569594604574, 17367.844925398746, 18907.486147260293, 20513.009833742948, 22198.763698197505, 23982.97418109085, 25881.587694134138, 27894.19767954961, 30006.8014146483, 32230.891605402365, 34599.759435588145, 37153.014318403264, 39927.116261881245, 42943.275299303175, 46237.38949927651, 49897.12288788886, 54028.86704695099, 58727.19477519617, 64034.59861652122, 69924.06581859202, 76403.8107635458, 83491.59550515993, 91045.28471403565, 98658.81669820586, 105799.94058744886, 111896.83364123061],[389.5253623803238, 617.374254047638, 967.6353562902827, 1495.8086679928276, 2277.163037593772, 3409.4453832793624, 5014.124398881855, 7234.516129599431, 10226.692544456124, 14133.720618228657, 19045.394798579862, 24969.143361617855, 31827.86170504625, 39486.48186116927, 47776.667378822865, 56485.028293319934, 65343.67481194415, 74078.84694839272, 82467.54468348014, 90351.20659949919, 97600.65414448225, 104055.93858828975, 109504.78754813853, 113740.78664829608, 116658.25711881359, 118335.47672709089],[1341.143198595081, 2007.6708289015069, 2955.4160030594953, 4271.09975654362, 6050.991809095964, 8392.608346069006, 11383.1001242266, 15085.79446446893, 19527.91202102824, 24692.924284133205, 30520.009550096467, 36910.84808753865, 43741.62300636484, 50876.724530846404, 58180.76824316427, 65526.7314435935, 72799.47219771569, 79894.98523426845, 86716.27734806525, 93166.9137071051, 99143.53816112255, 104529.5947075235, 109194.62537557432, 113006.54097952023, 115864.72869387119, 117752.0812456136],[11225.313721156663, 12513.047503775942, 13919.258503224508, 15457.612338358575, 17125.14395030098, 18896.54148895336, 20759.478456548342, 22729.44404683666, 24830.865456767122, 27088.22589226665, 29510.898863146558, 32095.310902970672, 34870.257948913124, 37893.46197709967, 41229.27750135125, 44931.972466825915, 49061.67746963264, 53733.04831602779, 59077.88529203694, 65184.33998253143, 72033.21070025227, 79542.8302956683, 87618.7405527538, 95951.39501530715, 103906.06755463018, 110754.11419404636],[194.01971488228287, 317.34625566838326, 515.058649265331, 826.799550875252, 1310.1240754018559, 2045.2006453717709, 3138.500761480325, 4723.977447637423, 6959.435293770232, 10015.967863889997, 14059.265719382718, 19218.003221648516, 25532.322839565215, 32915.12889010462, 41174.626678476736, 50073.22439990728, 59362.019677931006, 68763.66239428594, 77974.46955792722, 86722.33184230824, 94793.93236747116, 101999.1083305616, 108119.5865150356, 112920.15246296139, 116253.96120321075, 118184.13606803915],[682.8240079908859, 1065.794910338379, 1639.1314939577899, 2478.741939796816, 3678.690143440109, 5347.773004196466, 7601.363595439459, 10548.081689825558, 14272.888151332529, 18820.36084685968, 24183.06745012486, 30298.974152488336, 37058.78921240015, 44320.57345729362, 51926.78382322843, 59719.01599391279, 67547.46377577854, 75274.22388620056, 82771.09959526442, 89913.3246327031, 96571.1090565318, 102601.91870411235, 107848.74385884925, 112153.04538407385, 115391.77072023661, 117536.87234282025],[7811.186056674032, 8937.77771123705, 10176.224385553145, 11524.437033316686, 12998.148656029554, 14619.066107096885, 16404.482807796623, 18354.11760182843, 20444.846621535722, 22671.576183361976, 25062.061114025717, 27652.940600890266, 30473.172445381435, 33535.14485202077, 36879.85112239836, 40592.46423514624, 44768.03009350297, 49489.87415935172, 54864.270478695165, 61036.92890322453, 68102.60344053475, 76004.29350779658, 84548.17069720564, 93451.89455439826, 102121.70425247896, 109675.20736602595],[107.46979064348386, 178.6413871611937, 295.9928439266654, 487.05396605074526, 793.9686664655135, 1279.0367032356203, 2030.970036350855, 3170.436489097531, 4851.715767174175, 7256.445755013112, 10577.399333748659, 14991.432432923275, 20625.19084340162, 27515.89563508977, 35564.605825248815, 44527.63324196782, 54091.926787862874, 63947.665994789655, 73786.9342593256, 83281.94913780477, 92122.14680928674, 100043.14175497027, 106795.56169715355, 112126.83958766976, 115858.84476928892, 118034.8838556027],[369.3610704496076, 594.2979737860157, 944.9674467160457, 1481.2975058311622, 2283.907340635848, 3455.3339402161664, 5117.296723858469, 7401.942514431133, 10436.26338166403, 14321.399623972151, 19111.62293457915, 24799.584478798006, 31313.136086998886, 38524.76306175893, 46269.76279102413, 54366.60585577984, 62633.46834979905, 70897.56270711112, 78996.72115432634, 86774.59295248799, 94071.92160458725, 100717.63421995504, 106526.0359294041, 111310.22341621481, 114922.78284354416, 117322.739713349],[5356.577042943725, 6282.598310913933, 7322.932384533631, 8495.366780458146, 9812.56797411198, 11274.725282129382, 12885.376839757864, 14666.978799340623, 16644.647246388024, 18831.043131369242, 21213.478917395456, 23783.997816477207, 26577.61614372115, 29646.479021508316, 33034.77578254428, 36779.99297337298, 40968.90638681475, 45728.01380906784, 51175.02864216149, 57422.527475320916, 64606.631383690124, 72775.82282400077, 81751.80716996474, 91162.1948435336, 100440.40792743542, 108651.37470273992],[65.98088110301134, 110.27997338658446, 184.3644181077733, 307.24595415627766, 509.3338742526091, 837.8611424614077, 1363.788608104243, 2189.434742244106, 3455.0140500555426, 5340.670259853119, 8057.924798178788, 11824.259011372495, 16823.2932558448, 23160.08270945058, 30824.644178918337, 39666.31740594995, 49384.12142602487, 59594.902852028135, 69930.17008678429, 80042.01923889921, 89570.53389267456, 98168.04540880141, 105524.64774571861, 111359.75545518422, 115472.70193994469, 117887.68957607551],[213.84217842949084, 350.9812615407336, 571.2269078497916, 919.5039596064097, 1460.2160396019628, 2281.4750438308824, 3497.1109452256724, 5243.82067171668, 7670.821182786234, 10920.901406605666, 15105.189483043556, 20278.080270374165, 26420.865897792944, 33440.303907793685, 41182.24875370565, 49454.298785986975, 58048.87659796443, 66759.89443876983, 75389.98415105925, 83748.75338103372, 91644.80438577088, 98876.10703741174, 105226.20698960841, 110477.96556101559, 114457.73544367618, 117109.67830724735],[3584.4422788210723, 4348.516522252506, 5222.204594821686, 6214.449857215182, 7338.177732636875, 8614.834353323802, 10065.200420512592, 11699.093864252649, 13520.575211387519, 15552.872903285966, 17827.082500857654, 20360.82466379812, 23147.416241864965, 26198.179889841547, 29573.633419884, 33346.505942529984, 37579.219711883015, 42366.14894713375, 47858.69602149253, 54198.78552381891, 61499.036011345604, 69851.1352782067, 79168.99753426612, 89051.59413202363, 98864.53949545992, 107672.13868908989],[44.157753963103204, 73.8676834132557, 123.84724034674767, 207.45555161718298, 346.65720652251093, 576.8206111844527, 953.4870983979408, 1560.938260317737, 2521.4030798596414, 4002.1274489503994, 6215.4015520527755, 9404.638532502135, 13808.288227230803, 19602.419785462545, 26844.64733445739, 35443.54432205908, 45159.11682809451, 55616.903295140786, 66373.03377156174, 77011.40099242893, 87141.97488844668, 96364.86044263274, 104298.68780887319, 110616.91344915071, 115095.3481855161, 117742.52325794137],[132.5662360955084, 220.1104185432534, 363.56027782063677, 595.8925226550558, 966.7608234247228, 1548.0830934024968, 2439.1301422021224, 3768.753307170663, 5691.422221570096, 8373.802896639967, 11970.89309631247, 16595.486189843654, 22289.901476238596, 29010.538094134543, 36631.310916900125, 44963.566996063615, 53783.29308905166, 62855.251929881255, 71947.36113910879, 80833.7021895925, 89288.54317708538, 97076.69329061147, 103948.96144093816, 109656.16252631367, 113996.59900535742, 116897.68308957387],[2311.2727982539154, 2931.5797899880477, 3653.1390378545207, 4487.094595768942, 5450.2604385110035, 6557.565219653633, 7824.669888854085, 9277.191236422354, 10941.80751410256, 12833.802079129915, 14964.56865718014, 17366.515447309794, 20076.520505266497, 23111.430684245202, 26479.226064110364, 30240.324493640263, 34491.94491120005, 39326.883539449605, 44861.8085278401, 51268.6813053168, 58690.45067577449, 67205.76990619142, 76786.74389870446, 87084.97555554313, 97390.99097072524, 106736.82676031215],[31.613367675451144, 52.836289509509655, 88.62412322529046, 148.70530176983073, 249.3060652296156, 417.0998205975626, 695.2109072671367, 1151.6751372560561, 1890.0466302834336, 3060.1915194872704, 4864.796078099855, 7554.450881216317, 11402.739513440403, 16654.434075741032, 23450.280026057124, 31763.385812769582, 41387.782572062395, 51973.395930142855, 63066.02807519782, 74172.63521806365, 84839.71705714976, 94632.2850330779, 103112.51943277764, 109895.78985792426, 114726.52697026424, 117599.3580754222],[87.56277581394264, 146.23725521263853, 243.58406446551388, 403.77902902704034, 664.5928990446024, 1083.201302474804, 1742.7587462006895, 2758.069223155637, 4277.263794879734, 6475.238063672154, 9535.131675677, 13617.815005911354, 18825.881813949134, 25174.369611093185, 32580.063343738933, 40873.201670939925, 49824.79074660923, 59176.90395960571, 68664.98977214306, 78027.20903177102, 87001.88457854965, 95318.74039599468, 102694.00286517694, 108844.70526609516, 113539.34411978537, 116686.74904044086],[1430.6443097211816, 1905.5240951074431, 2485.109731673587, 3179.580799521104, 3995.9516245997393, 4947.90329964563, 6057.189235332687, 7344.487788366391, 8832.552757878242, 10554.634573237132, 12541.97022094318, 14813.074894900335, 17395.411015148966, 20336.879978560923, 23675.664162209534, 27434.150595788025, 31680.827162539408, 36537.950818177036, 42127.49357443264, 48594.07464753773, 56115.49012505225, 64790.787219694066, 74594.46441890854, 85242.86033685718, 96006.69312614494, 105843.56248267918],[23.844482983952467, 39.7898215057268, 66.70723266277784, 111.96136958688298, 187.92446750062953, 315.1616187390844, 527.4883335304439, 879.586806454528, 1457.6225304600675, 2392.159743213541, 3870.014373620873, 6137.818982285344, 9487.09087973696, 14211.38562217425, 20533.101360937704, 28513.798254881953, 37998.22705781885, 48641.27497206688, 59982.13448702849, 71496.42846458578, 82654.87788830957, 92969.9013732557, 101963.31344287551, 109194.02994729971, 114365.82570425949, 117458.1743283699],[61.140002221268176, 102.35367028748954, 171.21502322418684, 285.6523291021197, 474.43038508363077, 782.564783499486, 1278.2465008065737, 2060.3660414993687, 3264.2862333500198, 5061.683836986492, 7649.203202287294, 11222.34840325825, 15936.979423135555, 21869.190350834066, 28989.15835303146, 37159.9801544222, 46160.18102935737, 55717.45725820545, 65538.70922739037, 75326.92911316625, 84783.53967161317, 93601.58797712822, 101461.03420287235, 108043.48500342666, 113085.94148622805, 116476.87115505633],[858.386139171868, 1202.1919981803153, 1644.929761434552, 2198.1010288381617, 2875.324084808356, 3689.861929749563, 4653.4116136234325, 5786.903282501494, 7118.861168524404, 8676.648032905765, 10495.158939019275, 12616.830759888335, 15074.097095640573, 17897.66506021039, 21143.699727604966, 24869.078904375474, 29116.719120873593, 33977.6311527658, 39601.43379714265, 46134.393548351654, 53740.74175918898, 62559.524400543385, 72570.45673201059, 83517.78922247262, 94699.01974455197, 104992.97533358404],[18.73377027839279, 31.202768120204816, 52.26529307300881, 87.69778225037474, 147.23829447886584, 247.16543813607024, 414.50237546478655, 693.6113485448765, 1155.990663662493, 1913.5477392170014, 3133.7317134403106, 5050.737005960226, 7961.43278070153, 12192.528582293206, 18031.978333087758, 25632.916479058862, 34921.628609446256, 45575.87334925045, 57108.43615883087, 68967.00171753958, 80574.89619894601, 91376.40691837158, 100850.58572426347, 108510.06117931586, 114012.713869523, 117318.95284368074],[44.7433473069547, 74.94515053986478, 125.59448980760598, 210.2416679149121, 351.0261375457212, 583.4410231136062, 962.9440710877757, 1573.1370752970602, 2534.034728815083, 4006.9554146141645, 6190.423745014534, 9301.05668837904, 13537.355989550604, 19033.32648919743, 25818.165605624563, 33799.3229928474, 42775.289920924995, 52468.962155127854, 62564.097793785186, 72730.41558815519, 82632.18769022907, 91924.56878140985, 100249.75792885684, 107252.39325292724, 112636.36191358668, 116268.04444376496],[510.60221902912025, 746.2638799467331, 1066.4774381683267, 1489.7691301533314, 2032.3927353050767, 2708.766050665879, 3536.361892329588, 4532.1617063949025, 5716.379160673166, 7121.848977116903, 8784.780439018317, 10744.680898060764, 13052.759274020507, 15755.288924363997, 18893.52918940962, 22535.567238895375, 26756.62601869112, 31624.16260741762, 37264.24678113485, 43851.340355937595, 51542.972348617375, 60485.9857407023, 70691.37425665534, 81905.75320493901, 93461.19615870767, 104184.1037432802],[15.20465867940936, 25.27210647375665, 42.28642134439399, 70.91783865494565, 119.05399074978214, 199.9173487096987, 335.5699459298443, 562.5532920731396, 940.6198845395526, 1565.381822504222, 2584.5755988687215, 4214.403491706735, 6746.105221546959, 10526.00709096191, 15894.375194701368, 23085.97888731928, 32119.97520690709, 42731.58803187045, 54417.21244947249, 66574.13159084135, 78587.77586558007, 89846.11225361198, 99773.66589082393, 107842.87597865862, 113666.66026037573, 117181.66461080543],[34.04947177679233, 57.01091047346867, 95.58849261491137, 160.2573021133495, 268.33468477006943, 448.05138143493855, 744.535007781438, 1227.8789884755352, 2002.6052754483535, 3216.0281653453826, 5060.414089391829, 7761.747050504615, 11549.306146957097, 16607.921530613858, 23026.773336732855, 30765.91287363531, 39655.230494419964, 49423.01935244761, 59736.51104547446, 70235.1320689778, 80546.47968586157, 90287.00958637582, 99059.87622466036, 106471.32184277553, 112190.57632192587, 116060.26393208792],[306.55574932600956, 463.66120026042194, 687.9180423573664, 998.7252082479446, 1416.9482045465834, 1963.8216480324877, 2657.701795292318, 3518.154518134081, 4568.105335510204, 5831.84216486131, 7345.789970525831, 9155.498734143188, 11309.264893303047, 13868.07043287365, 16891.710080216068, 20436.657173732776, 24589.324032058794, 29444.220033154095, 35099.46925509144, 41721.9308765996, 49493.158953408536, 58550.7585905532, 68932.84688172716, 80395.04384118016, 92286.61091693751, 103412.35791215954],[12.668629588375534, 21.010115662799873, 35.11377004387988, 58.851954337738576, 98.77207799090273, 165.86627091448975, 278.5254070246181, 467.3647238276151, 782.8972447824355, 1307.1609694089218, 2169.8587180535083, 3567.2916751565153, 5776.899530880704, 9151.354129163132, 14070.292466391202, 20841.11356080132, 29573.856701329572, 40082.39538285022, 51878.47868699046, 64303.92926514101, 76686.84659443001, 88373.20543955074, 98731.54783051534, 107192.09011918007, 113327.22615149747, 117046.26063011424],[26.76619812170601, 44.777738196153535, 75.06562266206785, 125.91825219170278, 211.13729445114728, 353.4728738949419, 589.8846855448778, 979.0828786235076, 1611.2689615161546, 2618.557881356298, 4181.953672522941, 6527.636323800736, 9904.28680108481, 14538.250889102377, 20575.7747237514, 28034.253806601366, 36784.663933559386, 46570.885479491175, 57051.12004295887, 67838.46513589026, 78525.04216839437, 88688.23209342937, 97891.09114764452, 105700.16293624643, 111748.5557439695, 115853.52466076212],[188.33626503884466, 292.52182519856257, 446.8557795361209, 670.0361278052741, 984.0219266792183, 1412.3942524430533, 1980.1432436247576, 2710.90784011408, 3627.6616291100177, 4758.5969948520005, 6134.927482066806, 7799.214499960916, 9807.076107536132, 12218.59901065184, 15107.50926104902, 18547.887489556237, 22616.403495724604, 27423.737762997363, 33084.62809129628, 39735.974001358925, 47571.51451753686, 56738.98700277067, 67280.90705212357, 78974.70272083876, 91171.97609211609, 102673.54677936099],[10.784853698705662, 17.84422165851037, 29.78515174274061, 49.88630946403775, 83.6956710923246, 140.53547781446727, 236.02707700311214, 396.2518745377624, 664.4767485005718, 1111.6542859296117, 1851.7877830883597, 3061.7885425528693, 5001.074339560387, 8017.484410942821, 12515.6370477834, 18865.772002366622, 27265.87070716168, 37615.497288221464, 49470.428921149134, 62135.8966568159, 74863.61583782137, 86951.08541570412, 97721.00287675965, 106556.96466223852, 112994.01554337237, 116912.69268547017],[21.618066132113142, 36.124904958990804, 60.52980062827773, 101.53809008623354, 170.3630971389168, 285.619065913893, 477.88298327522136, 796.5278575523333, 1319.172304418793, 2163.0600458779227, 3495.3771846209856, 5535.981250683026, 8543.094279789744, 12774.519653592848, 18427.826778771243, 25579.155817265804, 34148.04232071701, 43903.575317293624, 54502.94903809253, 65537.73674234652, 76566.48070572819, 87127.5538069196, 96743.10479670357, 104938.80905277203, 111310.27132656352, 115647.82168577811],[119.27880207696035, 188.95443589370842, 295.3310537328682, 454.34816715047924, 686.3233320173093, 1015.8679238930779, 1470.0714314218008, 2077.628687870281, 2867.4671617769754, 3867.7220705098794, 5112.522232221585, 6641.715080702839, 8507.473115765742, 10777.441696970667, 13526.434877116157, 16844.29947150075, 20821.294924168917, 25561.110393148585, 31200.904080958746, 37876.587763581956, 45763.4533314967, 55030.897019242446, 65720.46897169997, 77629.94292352063, 90110.1369673471, 101961.54657608179],[9.34695219744561, 15.42775761654774, 25.717660651755757, 43.04177796310172, 72.18363911518831, 121.1854056168167, 203.53617152429996, 341.7979046473482, 573.5262416876956, 960.6927184423989, 1603.9752716202775, 2662.507870910622, 4376.370827621424, 7081.299606972177, 11193.046095463866, 17131.77435438756, 25179.563122505235, 35325.534396236595, 47185.58924149681, 60056.2546121473, 73110.85494501643, 85578.02864239209, 96740.84392822582, 105937.6173093518, 112666.86697758496, 116780.8782998692],[17.861845797368844, 29.809669445094965, 49.91397097937634, 83.71011812455363, 140.47781935734992, 235.6885624493032, 394.9438481341247, 660.065089506588, 1097.9440455965525, 1812.108283471819, 2955.096829522584, 4736.258492814248, 7415.454170475901, 11272.228369170349, 16547.98894572432, 23376.137187852037, 31729.828103167372, 41411.95887096142, 52086.91220363888, 63330.2164157021, 74669.38346581049, 85604.28889662004, 95615.61947438537, 104187.15308852229, 110875.69433210473, 115443.15007841724],[78.13897953837012, 125.57906035245094, 199.69062722558013, 313.4313176793255, 484.4170485237103, 735.2913145315831, 1093.737715543049, 1590.926800542486, 2259.9473838945287, 3135.2285615408596, 4251.439061272978, 5650.489007023974, 7382.863196607975, 9514.134562173145, 12126.863193865534, 15314.1967258784, 19185.056916028116, 23849.76991810593, 29446.010298611258, 36131.51204297138, 44064.48394261025, 53419.89720749771, 64250.16800780631, 76358.59911504724, 89103.9698703138, 101281.62608120967],[8.223412184112057, 13.539612483167073, 22.539271171930316, 37.692930118161826, 63.18602644913525, 106.05793700274495, 178.12315534153203, 299.16560422293605, 502.1893435285898, 841.8804870977601, 1407.7505711052181, 2343.160096145579, 3869.1531636684417, 6305.365046684852, 10067.95697120889, 15612.27211544677, 23296.83391731197, 33203.74470526669, 45019.11164642641, 58053.23462588302, 71415.50575834315, 84247.79955319906, 95787.20646713208, 105332.6988655979, 112345.50010648322, 116650.77548692253],[15.045708971857088, 25.074337721451876, 41.951353176423, 70.32897992153423, 118.01769358278777, 198.0738992260167, 332.20565296988093, 556.1598483892467, 927.8423672718536, 1538.5785975787128, 2526.539852192579, 4088.232767627339, 6479.254052901195, 9992.204364789646, 14904.06264644824, 21401.742304895834, 29514.686788835912, 39086.85179442934, 49797.84896085341, 61213.133161625214, 72832.32468588972, 84117.7490417783, 94508.33784529564, 103445.08833650169, 110444.79613993669, 115239.50492528793],[52.98980482958013, 86.0430391860153, 138.5614876020934, 220.85741366558102, 347.53062064880174, 538.5112792320926, 819.5634388708345, 1222.2159594184304, 1782.3819730417204, 2538.468467376738, 3531.186580762584, 4803.2612182763805, 6406.968505783889, 8406.309961239016, 10884.408645453908, 13942.173406490678, 17695.44127318428, 22273.890878371792, 27814.873763407682, 34488.99384030327, 42461.18346962491, 51893.3188761451, 62854.69518536228, 75149.53549728924, 88144.52123096536, 100625.74974102265],[7.327726738192361, 12.03446230261659, 20.005505039487744, 33.428658365591396, 56.01217291544927, 93.9947132335992, 157.85154173689483, 265.1379239192986, 445.18366433985295, 746.7239141767714, 1249.9478980890917, 2084.4991973515416, 3453.5568605339827, 5658.83663763492, 9109.517482798987, 14282.716657451025, 21601.584668538086, 31242.043044894865, 42969.65994799836, 56123.812791418946, 69770.01155776788, 82956.57187484531, 94858.91666979493, 104741.71917642713, 112029.83510866425, 116522.31275379247],[12.884355543978511, 21.43979724647095, 35.83868669630074, 60.0529507136517, 100.75683810410784, 169.12551438882605, 283.7960900082005, 475.6263918048908, 795.0619965008349, 1322.81308394393, 2183.601421161057, 3560.119052845771, 5699.598843117593, 8900.393854690865, 13466.761378464993, 19633.778078707997, 27487.65098777132, 36919.09801612742, 47630.55753735326, 59183.686989022295, 71053.86805493207, 82667.24425504025, 93420.96309066207, 102712.50850615879, 110017.54824771214, 115036.88132836085],[37.0972404038144, 60.754284897159195, 98.78048605676393, 159.26328771322923, 254.14146053548035, 400.3304540360598, 620.8875522566761, 945.6289623236812, 1410.9416832031181, 2058.380862176186, 2932.7991116718767, 4082.5479578950644, 5560.623389565063, 7432.974066856933, 9781.161185228555, 12710.230560158945, 16343.388433514532, 20822.92627361208, 26300.38987580327, 32946.9408232112, 40945.582154828844, 50447.37101637258, 61529.4566456109, 74000.79534352997, 87231.89512466238, 99998.50685019209],[6.601531873224534, 10.814212450277653, 17.951292456441212, 29.971311589760703, 50.1954148186504, 84.21235234271676, 141.40925352391582, 237.52690795318563, 398.89152032161843, 669.3342074770261, 1121.2441643778327, 1872.457443130938, 3109.901880261841, 5116.970260517775, 8291.020285472976, 13120.136756568198, 20078.736262926854, 29432.436733504554, 41035.13963939382, 54267.62022756011, 68169.15806150541, 81699.16272380648, 93954.20025869254, 104163.79530234392, 111719.72080586475, 116395.43989036893],[11.19148435193633, 18.592977554921507, 31.050331410067443, 52.001512112854805, 87.22686987567849, 146.41484356788652, 245.75518977660184, 412.1526771208122, 689.8777944025271, 1150.528371383783, 1906.6057520755, 3126.9458406800077, 5047.8167415333555, 7967.498553181502, 12209.74440658368, 18051.475525270158, 25634.255317640524, 34899.64375784071, 45579.826449975255, 57239.05998161451, 69332.56999635394, 81252.08368468312, 92353.19905894094, 101989.3077425042, 109593.92227272227, 114835.27440500338],[26.73976209219765, 44.09326574428453, 72.27672182270743, 117.60324863144042, 189.67783851559346, 302.6889106355155, 476.6911399771804, 738.8828134604126, 1124.2598272673852, 1675.2777358599274, 2440.3057573920914, 3472.1235298750125, 4828.729751768625, 6576.667471381178, 8799.622908895251, 11602.016379350087, 15114.746280946054, 19488.144456861308, 24891.125327424015, 31499.957440013324, 39508.81950974854, 49072.86303022361, 60264.39129404397, 72901.90769443245, 86355.11685873219, 99391.60038595076],[6.003370996159969, 9.809188491256558, 16.259378996068474, 27.123637202598225, 45.404128519768136, 76.15384628138469, 127.86227407831143, 214.7712763502015, 360.718946631796, 605.4516105511609, 1014.790716501861, 1696.4244680161794, 2822.7539703676257, 4659.366929870132, 7588.825555532344, 12101.827283343202, 18711.524650646676, 27764.957074353493, 39210.50535718395, 52482.80228222253, 66608.82376196118, 80469.85522996307, 93070.7475439023, 103598.05827316963, 111414.97379770153, 116270.11300613784],[9.84178328839672, 16.3232290943736, 27.23234856158575, 45.580833580385004, 76.43439452964779, 128.28952981789894, 215.3630494415747, 361.3404705786206, 605.3751900617306, 1011.2978157113926, 1680.717010272836, 2769.1600631673073, 4500.498828228117, 7168.524791040212, 11109.546778496351, 16635.585455810786, 23940.642872686185, 33019.6024672359, 43640.463671212245, 55376.42685210966, 67666.98283949, 79871.57639372596, 91304.75041236402, 101275.38064473614, 109173.88995319276, 114634.67928801285],[19.81462254855949, 32.82585763932846, 54.16142614017212, 88.81920591834287, 144.52020857556658, 232.97939464601131, 371.4352022219335, 584.1149812117224, 903.6052785745467, 1371.443659175418, 2037.5871515876975, 2958.7194848487134, 4197.587393006153, 5824.770840844607, 7924.9277125035305, 10604.437965923142, 13996.628614875217, 18261.317958101852, 23579.496035487067, 30142.134428673744, 38147.58411624887, 47764.58855451927, 59057.2774937635, 71852.61708052766, 85517.62108381704, 98812.08239770052],[5.504416932614866, 8.970938541849057, 14.848220899413983, 24.74844558300588, 41.40763438020945, 69.43162042319602, 116.56027700045162, 195.7824954917987, 328.85245009907374, 552.0819559843587, 925.7275961042832, 1548.7484754173952, 2580.68399047011, 4270.393093821504, 6984.14255727069, 11208.922047523796, 17485.617176905376, 26232.419301592334, 37493.65619652375, 50770.39885064696, 65089.27640436424, 79265.35126089043, 92206.38221362396, 103043.82580751873, 111115.38976806944, 116146.29256592924],[8.748728638401497, 14.485080517931431, 24.1402175332833, 40.38030931803278, 67.69126884479559, 113.60081947762443, 190.71640460655576, 320.0790952284906, 536.5850798745431, 897.4615483016896, 1494.7238859187391, 2471.4741631016614, 4038.620246433818, 6482.296638341879, 10145.435568950681, 15368.418437768027, 22393.644886305825, 31270.31048492023, 41807.32330109365, 53592.96492497703, 66055.65786952358, 78525.0321146276, 90275.32276933065, 100570.6222843704, 108757.42314954704, 114435.09112564918],[15.082577898090939, 25.03775151167012, 41.497512175692506, 68.48757017968165, 112.28428439909915, 182.55949169551127, 293.9158816478198, 467.68234191761815, 733.5612459937976, 1131.0496303669222, 1709.8524994081827, 2529.1335564035335, 3655.9021718900394, 5165.743248100305, 7146.143148357958, 9705.534026213187, 12979.118709745366, 17133.81250106336, 22360.897308184278, 28867.802662920785, 36859.46795968229, 46517.0397181155, 57902.86424064698, 70845.86180129151, 84711.4867551499, 98250.26714097106],[5.083103512824316, 8.263157906810832, 13.656676233123392, 22.742837591832238, 38.03286980812715, 63.75481979982007, 107.01497491368178, 179.7423817444277, 301.92599639073615, 506.9601297342741, 850.3475794601458, 1423.5070751458773, 2374.6217564181065, 3937.119891441434, 6460.535593546907, 10423.562025902313, 16385.36239185451, 24824.717507791916, 35879.369945108476, 49127.84318021874, 63608.719966666526, 78081.46680132381, 91357.34992360031, 102499.72431184608, 110820.61653069798, 116023.9857064707],[7.851175599713157, 12.975719006551735, 21.601082421047998, 36.10953252046404, 60.510273632010794, 101.53355163392831, 170.45881494791067, 286.13450407499835, 479.89443525466066, 803.3486805936745, 1340.1222083509385, 2221.9343619298716, 3646.7657124658886, 5890.9694804934525, 9299.21802102499, 14233.839479490598, 20980.835893105905, 29643.37353074269, 40075.32962725487, 51885.863503947636, 64497.14824693135, 77211.76197841484, 89264.6228425618, 99874.92822287368, 108344.49384563672, 114236.50508166636],[11.779787356119657, 19.551983955041518, 32.478481911227924, 53.84677138865836, 88.83615591503542, 145.5021114511714, 236.20014760944807, 379.45485780030083, 602.0314363259042, 940.7360082681042, 1443.805467048548, 2171.038844156996, 3192.874059825836, 4589.432647177185, 6453.089829234156, 8894.380611730936, 12051.682557594318, 16095.764010714654, 21228.394383881794, 27670.6210632344, 35640.36768971097, 45326.635662220346, 56798.02070361595, 69879.259031144, 83935.5826697557, 97706.99021321327],[4.723852245819247, 7.659725448859673, 12.640819825486478, 21.03292826596236, 35.155586863602934, 58.91461505580479, 98.87570223431597, 166.06304300749093, 278.95686706651264, 468.45269307002945, 785.964633436583, 1316.3717506151013, 2197.84300852033, 3649.7376878885534, 6005.135691795254, 9731.399875876705, 15398.004806484865, 23534.076564282455, 34365.60681210028, 47556.255298858, 62169.60385983067, 76918.80941708777, 90522.68035807599, 101965.56956678432, 110530.53902042555, 115903.12077045735],[7.105010404337742, 11.720963484721544, 19.490206390919376, 32.55887594293421, 54.53948878233918, 91.49811837730832, 153.60636929032006, 257.8774301374852, 432.64369786624064, 724.7239323760693, 1210.4287678578753, 2011.1799528048823, 3312.4653471315505, 5379.567914845202, 8555.022474071888, 13217.227482826202, 19690.567131756645, 28130.704315929106, 38439.498512934755, 50252.33259033427, 62990.01178882063, 75931.07921722517, 88272.3585729422, 99188.1945287981, 107935.07414993938, 114038.91633534322],[9.420437745076025, 15.612905588578876, 25.945034444117752, 43.12727954672273, 71.4839101311534, 117.80885327560928, 192.62304912872858, 311.96403608461605, 499.64800672738505, 789.593154859197, 1227.7178615577857, 1873.1702861573222, 2798.150216761113, 4087.029760641734, 5837.023819772549, 8162.811297093192, 11205.490324489809, 15139.730932191243, 20175.660169150542, 26546.671300661732, 34486.64289414612, 44191.814261781044, 55740.838376302076, 68952.73913177515, 83192.33093133676, 97187.69594311278],[4.414428533578761, 7.140093984228896, 11.766076574037971, 19.56054263532821, 32.6779376460252, 54.746514778648695, 91.86615712325809, 154.28100228375624, 259.16947087973466, 435.2674962961454, 730.4443472095384, 1223.8728153213583, 2044.8718342854438, 3400.040624278248, 5606.6681855918605, 9118.9696762944, 14510.279565766166, 22350.387109027026, 32947.03200272882, 46053.332336321226, 60770.97371256782, 75776.05752223116, 89700.06204413025, 101440.40505256344, 110244.90961002259, 115783.62839513663],[6.477810527318418, 10.666289777951384, 17.715896134814148, 29.57421257329969, 49.52006132806815, 83.06045810197463, 139.43338085445325, 234.10171366460855, 392.8504145377888, 658.3940718294699, 1100.6710567337614, 1831.86150763985, 3025.6366134719106, 4935.562024270055, 7899.068254877825, 12305.408628514342, 18511.981153409255, 26724.551765282562, 36894.956104592115, 48689.610927919406, 61532.813605857926, 74682.29983938375, 87298.23925898821, 98510.31779441817, 107529.13629672359, 113842.32008151313],[7.696131009069425, 12.727021772573657, 21.13369994867897, 35.164540500636214, 58.46023954252484, 96.81045776004576, 159.26548717428832, 259.7647507676693, 419.3851251491449, 669.0255620385734, 1051.9085924191027, 1625.4932473717847, 2462.3922570030936, 3650.0245317874055, 5290.368363816581, 7503.162694526153, 10432.90117871781, 14258.498780326412, 19196.20007081253, 25491.310353726534, 33393.021166705636, 43108.09596319581, 54724.78926839601, 68059.20274078389, 82473.12984105854, 96682.43282119717],[4.14561037669045, 6.688667490302128, 11.006129900795662, 18.281353812118738, 30.525332665900283, 51.125095170970795, 85.77560773727609, 144.0426370945033, 241.97166110854903, 406.4166665910604, 682.1502206772384, 1143.3355497343186, 1911.443902297617, 3181.5281365155274, 5255.948163888481, 8574.822673597824, 13710.360359425496, 21264.016943485898, 31617.987288547098, 44616.552975997125, 59411.40766017078, 74651.65517477249, 88886.76893191607, 100922.6115667214, 109963.3958181378, 115665.59594696164],[5.945344612630539, 9.77094690546966, 16.209614950968437, 27.04032199843328, 45.25842623024511, 75.89580100890569, 127.39621759878892, 213.901475677951, 359.01823820502005, 601.9275058068638, 1007.0090544836571, 1678.1878648959942, 2778.122611178996, 4548.488875799798, 7319.436436962799, 11486.571706309944, 17435.01064764162, 25417.522467425508, 35436.95489690035, 47194.9733593589, 60124.12858978045, 73464.74327626922, 86341.9756818876, 97841.19515186734, 107126.65264718179, 113646.71153059287],[6.408602067692561, 10.569158341930871, 17.52641221319545, 29.162264581369076, 48.5571426276799, 80.68216027879039, 133.39418849782484, 218.90199835429092, 355.88414176840774, 572.2801821698545, 908.3620334466676, 1419.2989497696055, 2176.9671387561, 3270.490034520345, 4806.098039452392, 6908.636904328205, 9727.4062092921, 13445.338825275443, 18284.45455833898, 24500.561659182076, 32357.098019220673, 42074.560382849726, 53749.56234972406, 67199.07601162414, 81779.00072172847, 96193.67076151617],[3.9105569097629953, 6.294051605934299, 10.341869430742726, 17.163246200761254, 28.643775631155812, 47.95958556160773, 80.45155077105312, 135.09200631729453, 226.93473544213026, 381.18459279715387, 639.8950980193326, 1072.814005287576, 1794.4406985408907, 2989.4044100600954, 4946.118046959251, 8090.288760048659, 12989.346559857153, 20268.381392625248, 30376.123815677878, 43247.54903330906, 58094.04052553686, 73548.54328852877, 88083.69883534856, 100412.06826124845, 109685.90163331007, 115548.837934838],[5.489227124740871, 9.00401405353707, 14.91935537065195, 24.869761586982335, 41.607652765285835, 69.75752183063392, 117.08170918665456, 196.58701942741754, 330.0038281487539, 553.453763412613, 926.4551989885751, 1545.5755249448916, 2563.314573794458, 4209.621097733048, 6805.849732866991, 10750.172133304945, 16450.36440743751, 24202.5950633595, 34060.887232565125, 45765.73748765484, 58762.54374773432, 72277.73300035765, 85403.28022606732, 97180.72428877652, 106727.59569053102, 113452.0859086107],[5.4284816545154415, 8.92503020183483, 14.773401467928547, 24.56751172610907, 40.93211526539956, 68.15386655057712, 113.10184556250795, 186.5546244619486, 305.14888324457985, 494.11682748015613, 790.6605100340828, 1247.2755829959096, 1934.26058792003, 2941.2775064511397, 4377.719101383651, 6373.488699244767, 9083.313376350663, 12694.644718492898, 17435.252593606783, 23569.767540025496, 31375.325833138722, 41087.71362631122, 52811.90857872804, 66368.97205970375, 81106.72932124129, 95717.72242911565],[3.7035286065923616, 5.946511814737686, 9.756849185440903, 16.17850418646255, 26.986616011958038, 45.171523092797116, 75.76211285477395, 127.20769437381784, 213.68755833184505, 358.9509878688983, 602.6475652750652, 1010.6091836022458, 1691.113785492398, 2819.367457333866, 4670.826587989954, 7656.865660486229, 12337.526440184016, 19354.670624066304, 29215.388501450758, 41943.458743755415, 56817.70869879486, 72465.99349472215, 87289.39751785772, 99907.24035552786, 109412.04484865318, 115433.43490160366],[5.0953150232409925, 8.341704819265727, 13.805107431493592, 22.99525418689878, 38.4546755231366, 64.45578087989344, 108.17162855695356, 181.62647892535102, 304.9232176236521, 511.5193726236977, 856.6669179184953, 1430.3783654663907, 2375.845967387645, 3911.6811488385547, 6349.46683355828, 10086.830137099216, 15549.503164133428, 23073.128338443836, 32762.296347471703, 44399.269644186425, 57446.66038089515, 71120.59711395956, 84481.86699525561, 96528.8034634135, 106331.93804508125, 113258.43845723332],[4.669051883036954, 7.650342762375975, 12.636575260565477, 20.994237612407264, 34.98135458202165, 58.313642160166005, 97.02054984908311, 160.68076079612646, 264.2081726824393, 430.4428333317209, 693.61783635308, 1103.256293033698, 1727.5300461749434, 2655.6233779069003, 3998.97918737719, 5892.058316435185, 8495.149892285383, 12001.15447165651, 16643.439563891574, 22694.601322712308, 30444.667530888833, 40145.074676787706, 51911.2484296449, 65569.78515695206, 80460.0725926183, 95262.01400639619],[3.519838502055097, 5.6382480914057265, 9.23799008965472, 15.305148442293811, 25.516896630374678, 42.69876929858896, 71.60286096300294, 120.21433234544642, 201.9360745069586, 339.22405732585077, 569.5888695820865, 955.3690052489267, 1599.2644738604615, 2667.9438628156377, 4424.860838284278, 7267.388556779053, 11746.360308068342, 18514.716381076687, 28129.81232317744, 40701.06755166108, 55580.86819216049, 71403.02233402006, 86503.01655953305, 99407.29556284576, 109141.54511796626, 115319.20035336357],[4.752591562379855, 7.765490357335748, 12.835703176230023, 21.36438633169122, 35.711396737301186, 59.842624060737755, 100.41785004638919, 168.6047804530234, 283.08526507417236, 474.98360929506896, 795.7925882418251, 1329.6797240766605, 2211.3462660492814, 3648.597968685822, 5942.694220706327, 9488.227394426942, 14724.607766592664, 22022.863805185167, 31536.885098148523, 43092.990171957426, 56175.09610597135, 69992.66890829484, 83577.4519240134, 95885.33151932515, 105939.65245927231, 113065.76443379215],[4.071189626700591, 6.6464509670238225, 10.952324990485378, 18.173901505339867, 30.27361140616805, 50.496561339340786, 84.15490508100267, 139.79590977188064, 230.86859917864092, 378.1409429911097, 613.1054244742611, 982.2054693425289, 1551.0415927951192, 2407.562663002042, 3664.2158580136356, 5459.141029540117, 7958.166877884847, 11360.235554182302, 15904.556625250836, 21871.469568371183, 29562.77777160894, 39245.16530041773, 51046.246147855636, 64798.42046478213, 79833.78259007641, 94819.16649783576],[3.356190915600502, 5.363627174675712, 8.775745699302272, 14.527071758803043, 24.207493015009224, 40.49569161785735, 67.89708465869512, 113.98308927483797, 191.46424044644908, 321.64241523064675, 540.1171699820436, 906.0989861916261, 1517.2726634798548, 2532.5661108029517, 4204.350054184147, 6916.51394336207, 11209.499885440146, 17742.70676392272, 27116.04540390474, 39520.119975908405, 54385.36136336575, 70361.67650987409, 85725.54242337376, 98911.76679098821, 108874.15617231358, 115206.27781535476],[4.4523761796946575, 7.260770761464302, 11.986582653649286, 19.935854052558348, 33.30838368795773, 55.80142073766528, 93.62470954821276, 157.19438893851887, 263.9438407555033, 442.9425327618341, 742.3567747092886, 1241.1325624193662, 2066.2437223717893, 3415.301710930005, 5579.016777137129, 8947.005357816983, 13968.541885036078, 21045.923559402658, 30380.522528664937, 41844.37803909828, 54946.48671927788, 68893.2873926689, 82689.75288471955, 95250.2078994835, 105550.71181267878, 112874.05911130866],[3.593799172727988, 5.844666504664084, 9.60635878840228, 15.917749382816963, 26.501328337269833, 44.215253482397856, 73.76600066860966, 122.80211282654656, 203.5052020409544, 334.85282292757046, 545.8799222514359, 880.0265730432976, 1399.99417386615, 2191.9203375197276, 3368.332131232974, 5070.097931166337, 7468.22072108109, 10767.864631562894, 15214.603299549126, 21096.26167633166, 28725.296093470977, 38383.23378653341, 50211.65833334019, 64049.245472876035, 79222.50214239012, 94384.12745366262],[3.209328512612015, 5.117278832618806, 8.361146408445542, 13.829223107433291, 23.033110420697945, 38.51976452908498, 64.57330461126847, 108.39389509560198, 182.07060296567153, 305.8687621280144, 513.6697593329068, 861.8664000054766, 1443.609757857243, 2410.7819179352355, 4005.5131247063664, 6598.796332270122, 10719.942727685051, 17031.092094278993, 26167.704983034462, 38396.29823087706, 53228.628364860575, 69340.19394571289, 84955.706735826, 98419.64629927007, 108609.42138912939, 115094.36480541446],[4.187756609646485, 6.815920086734959, 11.238188038114556, 18.676763833395935, 31.19033630175595, 52.239264191231996, 87.63629774787333, 147.13417884652802, 247.06309200119227, 414.673184805491, 695.1743497286593, 1162.8365174700216, 1937.6079071095987, 3207.551909382054, 5252.847313480379, 8456.667595339379, 13274.811112299489, 20136.80416588718, 29289.24844826878, 40650.974803205965, 53759.487904074864, 67821.79779363103, 81818.48979000228, 94623.33265993695, 105165.08911698373, 112683.31777851871],[3.2074427761386035, 5.195697085941099, 8.516452974283766, 14.089422995645858, 23.44053209243783, 39.10824245109867, 65.2904092437727, 108.85751662078468, 180.8715717090998, 298.7501481533114, 489.35488008657825, 793.3088869594911, 1270.2712871633933, 2004.0774742311296, 3106.6083280809903, 4720.543102641887, 7021.416758975798, 10220.67813395574, 14570.79911391395, 20367.1386493225, 27932.25061887832, 37561.84434544558, 49412.231408425345, 63329.23452504963, 78635.22274522156, 93967.31235554334],[3.077092864180531, 4.89547639375896, 7.987849387065383, 13.200882282197322, 21.97568353215382, 36.74058566020475, 61.58039962973133, 103.36086672857152, 173.61106620299, 291.66184424942315, 489.84418355856644, 822.0041946429153, 1377.1827197972038, 2300.8369521076165, 3825.6447407183678, 6310.350133878024, 10272.75532641887, 16374.795163578507, 25281.20899754682, 37328.48508282923, 52111.48819970114, 68339.96079269328, 84194.48242304007, 97930.94416531327, 108347.25987295965, 114983.71629310687],[3.95317494813105, 6.4215902346462235, 10.574792964563526, 17.560664800805178, 29.312781979884136, 49.08142759640384, 82.32718267565177, 138.21397863861958, 232.0919285024176, 389.5922471718409, 653.2858836428493, 1093.2431662861413, 1823.0242522796339, 3021.7944809423925, 4959.394317071726, 8011.487716077598, 12637.520031502461, 19290.36728687369, 28259.276199850978, 39510.38795182401, 52612.7767828022, 66777.55202410575, 80963.38469061843, 94004.60648296907, 104782.75751692071, 112493.53573989647],[2.891105019465359, 4.664328696323817, 7.6237690274508205, 12.591071625797225, 20.92972960773123, 34.912360970145784, 58.30933261396015, 97.32345125269616, 162.02561155781504, 268.4479070719755, 441.5341371580924, 719.3302243890594, 1158.4674496800628, 1840.0962302341425, 2874.8346515197704, 4406.308966118434, 6613.783323814948, 9714.77697285088, 13968.911840884724, 19679.237486718233, 27178.032596333444, 36774.42808693578, 48639.71202744992, 62628.02669339795, 78059.38918309398, 93555.5042539547],[2.9575725016895245, 4.695066224882052, 7.650588556770041, 12.63321425322562, 21.020365152079357, 35.1331940164379, 58.87641235520326, 98.81351926329096, 165.9673539249773, 278.8235746483621, 468.3097977847776, 785.9636991107033, 1317.0906110940844, 2201.279152393393, 3662.4858087241296, 6047.883212640256, 9863.651794408193, 15769.227385058599, 24453.115046314797, 36315.542913285986, 51034.77723576513, 67362.56987125729, 83443.10583115775, 97445.97762332618, 108087.41854765441, 114874.08868759141],[3.74412199678979, 6.070196915775293, 9.983638011635437, 16.566095375552656, 27.63963408799892, 46.2672673928852, 77.5955501561659, 130.26312807185207, 218.74507471164904, 367.2251384044706, 615.9087447832642, 1031.0828378973838, 1720.494183052137, 2855.04326416977, 4694.546722050539, 7606.423849416522, 12051.328531806379, 18501.827712299302, 27286.99380329699, 38420.29364826289, 51505.053316690486, 65759.90912259625, 80124.16186879021, 93393.93068976777, 104403.69029118551, 112304.70831567672],[2.629156341058781, 4.224345308725096, 6.884439417705985, 11.349571665501061, 18.84778025765179, 31.42890589037382, 52.50226288314952, 87.6989158113532, 146.21900039781468, 242.84781628643933, 400.8203170605062, 655.8539718997822, 1061.6737117357366, 1696.5219151962835, 2669.20923457457, 4123.57840826465, 6241.8132038191525, 9247.061288201554, 13406.217282130216, 19030.391681440964, 26461.433301511857, 36021.16765294924, 47895.91154815131, 61949.53339583102, 77501.43681618905, 93156.05916132298],[2.8489325072748546, 4.51297280422904, 7.344191750528717, 12.117516667847214, 20.15251419309495, 33.67296253191184, 56.419933930629426, 94.68226779363785, 159.02264904763226, 267.1581706888635, 448.73940198182476, 753.2007559990388, 1262.4362889675813, 2110.651919170763, 3513.73575967295, 5807.9453770037535, 9487.889543977539, 15208.730389522583, 23677.982158487575, 35353.520950607926, 49996.1549170167, 66406.69725751913, 82700.5997348821, 96963.78622437945, 107829.20560293074, 114765.28705712172],[3.556908838744329, 5.755535486245694, 9.45428457395223, 15.675496961599547, 26.14136675988828, 43.74715996533139, 73.3580720036358, 123.14190920688637, 206.78884930909777, 347.1827054610898, 582.3998476177003, 975.3078782042085, 1628.3556140780793, 2704.782212317895, 4454.774219049828, 7237.0401752308435, 11511.408391300232, 17766.73938915551, 26368.963657611916, 37378.438915071354, 50435.041556000026, 64768.2356626547, 79300.54792701619, 92791.20725260675, 104027.86085331594, 112116.83084187697],[2.410073335084554, 3.856384586787659, 6.266029106437025, 10.310764623530641, 17.104716721496874, 28.50965149883237, 47.62819846922645, 79.60090959440939, 132.8672624412093, 221.0922582804883, 365.96299420329865, 601.0996960756187, 977.5033066438801, 1570.4318228868603, 2486.4390276478116, 3868.95273194573, 5902.303784231938, 8814.696171516254, 12880.292901499652, 18418.603607348763, 25781.012521113837, 35301.56995937454, 47181.536816569576, 61295.70154948966, 76962.52608544876, 92771.89018965398],[2.749776259265196, 4.346766836700247, 7.064507692109478, 11.646761300681318, 19.360278137468995, 32.339936343861986, 54.17739855674, 90.91069537741916, 152.68222312211185, 256.50683067971227, 430.86749279694163, 723.2734900882771, 1212.490242585884, 2027.7680141745166, 3377.5105786857325, 5587.684977715791, 9141.493003611191, 14688.451216065085, 22950.978619036327, 34438.74537544907, 48993.26463220255, 65470.77749389161, 81965.88233604758, 96483.85428778722, 107572.60496076668, 114657.67426055387],[3.388494220118834, 5.472490631635029, 8.978126554273636, 14.874392907051831, 24.793635221658644, 41.48018554044042, 69.54602445038937, 116.73506011368633, 196.0303832966361, 329.1434855829106, 552.227078679459, 925.0484899562575, 1545.2196005258877, 2568.884828673239, 4237.041504536721, 6899.435646814258, 11013.400912690606, 17080.97998182734, 25501.92098204708, 36382.64328982961, 49401.49074482123, 63801.906132911114, 78492.27187238206, 92196.33880654261, 103655.24275254199, 111929.89867031846],[2.225300643177391, 3.5460936223190886, 5.744480445674225, 9.434433287596356, 15.6336018925054, 26.04397491844893, 43.50636681594677, 72.73935141423837, 121.51959296334338, 202.51340347602715, 335.9962853911633, 553.6886771319766, 904.0809146448713, 1459.4873198088671, 2323.8891977740527, 3639.726824861581, 5592.771018128313, 8415.610361187324, 12389.441396348217, 17842.39175958252, 25135.53103532928, 34614.777464889194, 46495.911799431415, 60665.38978001872, 76441.62112653843, 92399.3373057783],[2.6591425410113416, 4.194913313490104, 6.809013337592959, 11.216742797817925, 18.636608039360993, 31.122276347137607, 52.12891237387614, 87.4653840289044, 146.88999434381788, 246.77560697656926, 414.53717594403986, 695.9212758519222, 1166.8235788938687, 1951.9335393168149, 3252.7212849823445, 5385.488560942147, 8822.326190868536, 14206.09825538274, 22270.607683220296, 33571.68437533175, 48028.61765496328, 64558.138227956086, 81241.7381047784, 96007.57876499447, 107317.65355492805, 114550.89616272705],[3.2363527452246714, 5.216814688491495, 8.548017487931213, 14.150761509201997, 23.576222065445986, 39.43235891801578, 66.10232201988498, 110.9468152928721, 186.3093655670338, 312.8400880506948, 524.9472228727477, 879.5782022657112, 1469.9197789742539, 2445.5479017917473, 4038.7348628258105, 6590.179790177085, 10553.376195221506, 16440.734429050197, 24682.771169961354, 35430.79998959794, 48403.17628497277, 62860.303288037445, 77699.06519640236, 91609.22866063204, 103285.8096746052, 111743.90716864623],[2.067928920904936, 3.2818749505548013, 5.300339303629285, 8.688010319550559, 14.380102765249466, 23.941736433686295, 39.98855039616016, 66.87390604267341, 111.79510077231683, 186.530474506021, 310.06770399527676, 512.3797599871564, 839.6493271400544, 1361.3508983250413, 2178.696166096906, 3432.6121358657374, 5309.652833841274, 8046.079467860761, 11929.70747520077, 17297.31938055428, 24519.791304261078, 33954.55026197384, 45831.41769000796, 60049.44716847385, 75928.64717113836, 92029.65805056451],[2.575807546009936, 4.05534382322947, 6.574217114632826, 10.821577161189982, 17.971597718313582, 30.00331398160163, 50.246440612823335, 84.29920745978708, 141.5668080882166, 237.83172621887599, 399.5262340111945, 670.773596143515, 1124.822511016011, 1882.1433153998091, 3137.754666178991, 5198.857027007614, 8526.759418197786, 13756.919999882623, 21631.55749766762, 32747.534220240712, 47098.620789564215, 63666.19417651257, 80526.254267361, 95533.50033569548, 107063.3958978446, 114444.69958537139],[3.0983732587307324, 4.984956733589967, 8.157983051080791, 13.49455256415732, 22.47222468069691, 37.575264884843484, 62.9792291704171, 105.69707741604574, 177.49166507072272, 298.0486492908401, 500.18878849157545, 838.2867342785213, 1401.4719248206707, 2333.2370404751136, 3857.5995391420406, 6306.255284116118, 10127.794454906196, 15842.477898269866, 23908.586220443314, 34520.876621376774, 47438.90056404877, 61942.81847110354, 76920.66195043208, 91029.78080867085, 102919.53544254764, 111558.85172034855],[1.9328638088440477, 3.0551646781460877, 4.919231742124392, 8.047417072319412, 13.304008829551556, 22.136123488978104, 36.96462854731128, 61.82535460601207, 103.40784103075423, 172.70215183825104, 287.5276609361465, 476.2395922095438, 782.8923446908407, 1274.2698174191837, 2048.7309752263013, 3245.2587047483885, 5050.592693162242, 7704.001377431729, 11499.440338204087, 16782.29495052811, 23933.509325999603, 33321.977636115735, 45191.28056298939, 59453.271737104325, 75430.34727873866, 91669.85225950652],[2.499088966789204, 3.926875852780368, 6.3581009474952825, 10.457849642656772, 17.35948987875807, 28.973357177900976, 48.513680665105305, 81.38476368277833, 136.66665096547223, 229.59804184747293, 385.7056529108316, 647.6156275641599, 1086.1319830372936, 1817.8182108144442, 3031.688328035234, 5026.381919641057, 8252.807748999752, 13338.49689979518, 21031.554416237734, 31965.110831221493, 46203.63871501347, 62796.1629674292, 79820.75458060364, 95062.69894942714, 106810.44282261332, 114339.41527709048],[2.9727797905422375, 4.773929619469082, 7.802997391901384, 12.897311221874293, 21.467422983933037, 35.88499625094825, 60.136578967241476, 100.91843597506411, 169.4643598989607, 284.580600327924, 477.6385476791671, 800.6585561167778, 1339.0415115803444, 2230.641908278283, 3691.686471031832, 6045.0069088264045, 9733.469663117954, 15282.958476453108, 23176.60040235841, 33650.91547637772, 46507.49365310421, 61048.85190785847, 76156.79881669379, 90457.89993946065, 102556.3940174714, 111374.72772477547],[1.8162515498739202, 2.8594743547760597, 4.590262617182926, 7.494391187435486, 12.374792005611098, 20.57632923834901, 34.350645950794004, 57.456517512385, 96.13752786756689, 160.68435701503844, 267.86224658166407, 444.53231875319494, 732.7723935617131, 1196.8485203099958, 1932.28008279551, 3075.7964457361013, 4813.776574831433, 7387.878419382327, 11097.62140223349, 16296.840448661404, 23376.668577318513, 32717.396897686733, 44575.87478147852, 58877.150487897634, 74947.50510525922, 91321.86587216385],[2.428249011018152, 3.8082758784884554, 6.158593291946151, 10.122076870502589, 16.79442516376636, 28.022550224282597, 46.9140586545098, 78.6941939191365, 132.14273244874164, 221.99607087160413, 372.9440857510855, 626.228381181071, 1050.3892319140145, 1758.3640413620628, 2933.568526295367, 4866.5858557726915, 7998.324954529733, 12948.054968107803, 20467.637540469106, 31222.15287739114, 45342.81069785573, 61948.25124511222, 79125.7910436367, 94595.52277912662, 106558.77643283598, 114234.95857563766],[2.858069531620104, 4.5812056534091745, 7.478807639975267, 12.351882848606458, 20.54978327577225, 34.341320413096746, 57.54037476596681, 96.55382606090768, 162.13183100739042, 272.27618872454, 457.03091382333974, 766.2558573119072, 1281.9176044606866, 2136.6394090446124, 3539.3070772281817, 5804.0963824938535, 9367.535658373521, 14759.179878933784, 22484.205293991196, 32819.03344583717, 45607.813879826, 60177.81297353476, 75407.21517497028, 89893.49144660667, 102196.35949926816, 111191.53059715626],[1.7148120250294898, 2.6892994799390904, 4.3041878815893915, 7.013426709583769, 11.566497620636763, 19.21906217249973, 32.074806347226335, 53.649402754461775, 89.79294395696802, 150.17375079119728, 250.6064148924936, 416.57559881964033, 688.3076785639375, 1127.7092904763513, 1827.5237221339178, 2922.003450193189, 4596.647534669075, 7094.896745455842, 10721.246689302316, 15837.684045620004, 22845.670801079417, 32137.035033512027, 43981.61958384894, 58317.59445716681, 74475.92638597582, 90980.12661875716],[2.3626441448065165, 3.698482581546032, 5.973923650836173, 9.811289486750884, 16.27141364518745, 27.142502650937246, 45.43346603553962, 76.20378284689356, 127.95521721436992, 214.95898351593647, 361.129650510693, 606.4252131110691, 1017.2847598266869, 1703.272912631637, 2842.577167930478, 4718.1939162801455, 7761.436113056355, 12583.11557734253, 19937.04583701725, 30516.322674988525, 44514.797083059515, 61121.77179113323, 78440.79653327604, 94131.4018609358, 106307.79740187421, 114130.93870575044]][:105]
    with open("heatmap-data3.txt", "r") as file:
        vaxmxstring = file.read().replace('\n', '')
    vaxmx = ast.literal_eval(vaxmxstring)
    print(len(vaxmx))
    firstones = []
    secondones = []
    thirdones = []
    for i in range(len(vaxmx)):
        for j in vaxmx[i]:
            if i % 3 == 0:
                # print(j)
                firstones.append(j)
            elif i% 3 == 1:
                secondones.append(j)
            else:
                thirdones.append(j)
        if i % 3 == 2:
            # print(i,len(firstones))
            for j in range(len(firstones)):
                # print(j,i//3)
                if firstones[j] > secondones[j] and firstones[j] > thirdones[j]:
                    vax_matrix_worst[i//3][j] = 0
                elif secondones[j] > firstones[j] and secondones[j] > thirdones[j]:
                    vax_matrix_worst[i//3][j] = 1
                elif thirdones[j] > firstones[j] and thirdones[j] > secondones[j]:
                    vax_matrix_worst[i//3][j] = 2
                # else:
                #     print("error",firstones[j],secondones[j],thirdones[j])
                # print(j,i//3)
                if firstones[j] < secondones[j] and firstones[j] < thirdones[j]:
                    vax_matrix[i//3][j] = 0
                elif secondones[j] < firstones[j] and secondones[j] < thirdones[j]:
                    vax_matrix[i//3][j] = 1
                elif thirdones[j] < firstones[j] and thirdones[j] < secondones[j]:
                    vax_matrix[i//3][j] = 2
                # else:
                #     print("error",firstones[j],secondones[j],thirdones[j])
            firstones = []
            secondones = []
            thirdones = []
########################

    #### (NEED FOR BOTH)
    def get_alpha_blend_cmap(cmap, alpha):
        cls = plt.get_cmap(cmap)(np.linspace(0,1,256))
        cls = (1-alpha) + alpha*cls
        return ListedColormap(cls)

    vax_matrix_columnsx = [1*i for i in range(numruns)]
    vax_matrix_columns = [format(1000*j,",") for j in range(1,numrunsperday+1)]
    myColors = (palette_five[1],palette_five[2],palette_five[4])
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    ####

######################## FOR POSTER
if format_heatmap_poster:
    vax_matrix_worst_df = pd.DataFrame(data=vax_matrix_worst, columns = vax_matrix_columnsx, index = vax_matrix_columns)
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(4,5),  gridspec_kw={"height_ratios":[0.035, 1]},constrained_layout=True)
    yticks = np.linspace(9, len(vax_matrix_columns) - 1, 18, dtype=int)
    yticklabels = [vax_matrix_columns[idx] for idx in yticks]
    sns.heatmap(vax_matrix_worst_df,  ax=ax, cmap=get_alpha_blend_cmap(cmap,0.75),vmin=0,vmax=2,xticklabels=10,cbar=False, rasterized=True)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    ax = plt.gca()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.3333, 1, 1.6667])
    colorbar.set_ticklabels(["Uniform strategy","Age-based strategy","Contact-based strategy"])
    colorbar.ax.tick_params(length=0)
    colorbar.outline.set_linewidth(0)
    plt.xlabel("Vaccination delay")
    plt.ylabel("Initial vaccination capacity (IVC)")
    # plt.suptitle("Least effective vaccination priority strategy")
    # fig.tight_layout()
    ax.invert_yaxis()
    print("worst:")
    plt.savefig('graphics/heatmap-prio-w-final-adj-poster.pdf', dpi=800, bbox_inches="tight", transparent=True)
    plt.show()

    vax_matrix_df = pd.DataFrame(data=vax_matrix, columns = vax_matrix_columnsx, index = vax_matrix_columns)
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(4,5),  gridspec_kw={"height_ratios":[0.035, 1]},constrained_layout=True)
    yticks = np.linspace(9, len(vax_matrix_columns)-1, 18, dtype=int)
    yticklabels = [vax_matrix_columns[idx] for idx in yticks]
    sns.heatmap(vax_matrix_df,  ax=ax, cmap=get_alpha_blend_cmap(cmap,0.75),vmin=0,vmax=2,xticklabels=10,cbar=False, rasterized=True)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    ax = plt.gca()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.3333, 1, 1.6667])
    colorbar.set_ticklabels(["Uniform strategy","Age-based strategy","Contact-based strategy"])
    colorbar.ax.tick_params(length=0)
    colorbar.outline.set_linewidth(0)
    plt.xlabel("Vaccination delay")
    plt.ylabel("Initial vaccination capacity (IVC)")
    # plt.suptitle("Most effective vaccination priority strategy")
    # fig.tight_layout()
    ax.invert_yaxis()
    print("best:")
    plt.savefig('graphics/heatmap-prio-b-final-adj-poster.pdf', dpi=800, bbox_inches="tight", transparent=True)
    plt.show()
########################

if format_heatmap_poster_2:
    fig_width_pt = textwidth
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * (1 / 2) / golden_ratio
    figsize = (fig_width_in, fig_height_in)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, constrained_layout=True, sharey=True)
    vax_matrix_df = pd.DataFrame(data=vax_matrix, columns = vax_matrix_columnsx, index = vax_matrix_columns)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(vax_matrix_df.iloc[150:250,145:155])
    sns.heatmap(vax_matrix_df,  ax=ax1, cmap=get_alpha_blend_cmap(cmap,0.75),vmin=0,vmax=2,xticklabels=50,cbar=False, rasterized=True)
    ax1.set_xlabel("Vaccination delay")
    ax1.set_title("(a) Best strategy.")
    vax_matrix_worst_df = pd.DataFrame(data=vax_matrix_worst, columns=vax_matrix_columnsx, index=vax_matrix_columns)
    sns.heatmap(vax_matrix_worst_df, ax=ax2, cmap=get_alpha_blend_cmap(cmap, 0.75), vmin=0, vmax=2, xticklabels=50,cbar=False, rasterized=True)
    ax2.set_xlabel("Vaccination delay")
    ax2.set_title("(b) Worst strategy.")
    yticks = np.linspace(9, len(vax_matrix_columns) - 1, 18, dtype=int)
    yticklabels = [vax_matrix_columns[idx] for idx in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_ylabel("Initial vaccination capacity (IVC)")
    ax1.tick_params(axis="y",direction="out", pad=-2)
    ax1.tick_params(axis="x",direction="out", pad=-2)
    ax2.tick_params(axis="x", direction="out", pad=-2)
    plt.draw()
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    cax = fig.add_axes([p0[0], -0.075, p1[2] - p0[0], 0.05])
    fig.colorbar(ax2.get_children()[0], cax=cax, orientation="horizontal")
    # ax = plt.gca()
    colorbar = ax2.collections[0].colorbar
    colorbar.set_ticks([0.3333, 1, 1.6667])
    colorbar.set_ticklabels(["Uniform strategy","Age-based strategy","Contact-based strategy"])
    colorbar.ax.tick_params(length=0,labelsize=10.95)
    colorbar.outline.set_linewidth(0)
    ax1.tick_params(axis='both', which='major')
    ax2.tick_params(axis='both', which='major')
    # plt.suptitle("Least effective vaccination priority strategy")
    # fig.tight_layout()
    ax1.invert_yaxis()
    # plt.savefig('graphics/heatmap-prio-w-final-adj-poster-2.pdf', dpi=800, bbox_inches="tight", transparent=True)
    plt.show()

######################## FOR PRES
if format_heatmap_pres:
    vax_matrix_worst_df = pd.DataFrame(data=vax_matrix_worst, columns = vax_matrix_columnsx, index = vax_matrix_columns)
    fig,  ax = plt.subplots(nrows=1, figsize=(4,5),constrained_layout=True)
    sns.heatmap(vax_matrix_worst_df,  ax=ax, cmap=get_alpha_blend_cmap(cmap,0.75),vmin=0,vmax=2,xticklabels=50,cbar=False, rasterized=True)
    # fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    ax = plt.gca()
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_ticks([0.3333, 1, 1.6667])
    # colorbar.set_ticklabels(["Uniform strategy","Age-based strategy","Contact-based strategy"])
    # colorbar.ax.tick_params(length=0)
    # colorbar.outline.set_linewidth(0)
    plt.xlabel("Number of days before vaccinations begin")
    plt.ylabel("Initial vaccination capacity (IVC)")
    # plt.suptitle("Least effective vaccination priority strategy")
    # fig.tight_layout()
    ax.invert_yaxis()
    print("worst:")
    plt.savefig('graphics/heatmap-prio-w-final-adj-poster-longer3.pdf', dpi=1200, bbox_inches="tight", transparent=True)
    plt.show()

    vax_matrix_df = pd.DataFrame(data=vax_matrix, columns = vax_matrix_columnsx, index = vax_matrix_columns)
    fig, ax = plt.subplots(nrows=1, figsize=(4,5),constrained_layout=True)
    sns.heatmap(vax_matrix_df,  ax=ax, cmap=get_alpha_blend_cmap(cmap,0.75),vmin=0,vmax=2,xticklabels=50,cbar=False, rasterized=True)
    # fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    ax = plt.gca()
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_ticks([0.3333, 1, 1.6667])
    # colorbar.set_ticklabels(["Uniform strategy","Age-based strategy","Contact-based strategy"])
    # colorbar.ax.tick_params(length=0)
    # colorbar.outline.set_linewidth(0)
    plt.xlabel("Number of days before vaccinations begin")
    plt.ylabel("Initial vaccination capacity (IVC)")
    # plt.suptitle("Most effective vaccination priority strategy")
    # fig.tight_layout()
    ax.invert_yaxis()
    print("best:")
    plt.savefig('graphics/heatmap-prio-b-final-adj-poster-longer3.pdf', dpi=1200, bbox_inches="tight", transparent=True)
    plt.show()
########################

if make_deaths_by_priority_ages:
    # priority order with working age order
    categories = ["No vaccine","Prioritising working age population", "Vaccinating oldest to youngest","Vaccinating by social contact matrix"]
    grouped_deaths = np.column_stack([run_model(vperday=0)[3].sum(axis=0),run_model(vperday=100000,priority=[4,8,5,9,6,10,7,11,3,2,1,0,17,16,15,14,13,12])[3].sum(axis=0),run_model(vperday=100000)[3].sum(axis=0),run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0)])
    plt.title("Cumulative deaths over time")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette(palette_five[::-1])
    plt.xlabel("Number of days")
    plt.ylabel("Total deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,palette_five[::-1],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages-cum.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()
    for i in range(len(grouped_deaths[-1:][0])):
        print(categories[i],"total deaths:",round(grouped_deaths[-1:][0][i]))

    # cum priority order with working age order
    categories = ["No vaccine","Prioritising working age population", "Vaccinating oldest to youngest","Vaccinating by social contact matrix"]
    grouped_deaths = np.column_stack([pop.cumulative_to_daily(run_model(vperday=0)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000,priority=[4,8,5,9,6,10,7,11,3,2,1,0,17,16,15,14,13,12])[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000)[3].sum(axis=0),"")[0],pop.cumulative_to_daily(run_model(vperday=100000,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0),"")[0]])
    plt.title("Daily deaths over time")
    df_deaths_vax = pd.DataFrame(grouped_deaths, columns=categories)
    sns.set_palette(palette_five[::-1])
    plt.xlabel("Number of days")
    plt.ylabel("Total deaths")
    data=df_deaths_vax
    shading(sns.lineplot(data=data,dashes=False,linewidth=4),data.shape[1],0.1,palette_five[::-1],True,ndays+1,loc="upper left")
    plt.savefig('graphics/priorities-ages.png', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()


#  vax start date (daily)
if make_vax_start_date_daily:
    grouping_key = ["Under 20","20--39","40--59","60--79","80+"]
    # plt.figure(figsize=(4,2.5)) # for poster
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3,1)))
    # plt.title("Total deaths with varying vaccination start dates (no priority)") # for pres
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(351):
            if i%10 == 0:
                print(i)
            x_data.append(i)
            deaths_datal.append(run_model(vfromday=i,use_priority=useprio_key[graphnum],priority=prio_key[graphnum])[3][:,-1])
        deaths_data=np.array(deaths_datal)
        # print(deaths_data[:,0])
        grouped_deaths = np.zeros((5,len(deaths_data[:,0])))
        # grouped_deaths[0] = np.sum(deaths_data[:,:5],axis=1)
        for i in range(0,4):
            grouped_deaths[i] = np.sum(deaths_data[:,4*i:4*i+4],axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:,16:],axis=1)
        # print(grouped_deaths)
        lives_saved = np.zeros(len(deaths_data[:,0]))
        novaxdeaths = sum(run_model(vperday=0)[3][:,-1])
        for i in range(len(deaths_data[:,0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:,i])
            # print(i, sum(grouped_deaths[:,i]))
        # print("lives saved:",lives_saved)
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([i for i in range(len(grouped_deaths[0])-1)],cumulative_to_daily(grouped_deaths[0]),cumulative_to_daily(grouped_deaths[1]),cumulative_to_daily(grouped_deaths[2]),cumulative_to_daily(grouped_deaths[3]),cumulative_to_daily(grouped_deaths[4]), edgecolor="none", alpha=0.9)
        # for num, i in enumerate(np.array(cumulative_to_daily(grouped_deaths[0]))+np.array(cumulative_to_daily(grouped_deaths[1]))+np.array(cumulative_to_daily(grouped_deaths[2]))+np.array(cumulative_to_daily(grouped_deaths[3]))+np.array(cumulative_to_daily(grouped_deaths[4]))):
        # for num, i in enumerate(np.array(cumulative_to_daily(grouped_deaths[4]))):
        #     print(num+200, i)
        ax[graphnum].set_xlim(right=len(deaths_data[:,0])-1)
        ax[graphnum].set_ylim(top=1000)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="upper right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Days since first infection")
        # ax[graphnum].plot([i for i in range(len(grouped_deaths[0]))], lives_saved, '--', color=palette_neutral[1])
        # ax = plt.gca()
        if graphnum == 1:
            ax[graphnum].set_ylabel("Deaths from single-day additional delay in vaccination")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x",which="major",bottom=True,color="lightgray",width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-startdates-stacked-age-prio3-daily.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax start date (cumulative)
if make_vax_start_date_cum:
    grouping_key = ["Under 20", "20–39", "40–59", "60–79", "80+"]
    # plt.figure(figsize=(4,2.5)) # for poster
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    # plt.title("Total deaths with varying vaccination start dates (no priority)") # for pres
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(301):
            if i % 10 == 0:
                print(i)
            x_data.append(i)
            deaths_datal.append(
                run_model(vfromday=i, use_priority=useprio_key[graphnum], priority=prio_key[graphnum])[3][:, -1])
        deaths_data = np.array(deaths_datal)
        # print(deaths_data[:,0])
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        # grouped_deaths[0] = np.sum(deaths_data[:,:5],axis=1)
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        # print(grouped_deaths)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        # print("lives saved:",lives_saved)
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
        # ax = plt.gca()
        if graphnum == 1:
            ax[graphnum].set_ylabel("Total number of deaths")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
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
    for i in range(301):
        if i % 10 == 0:
            print(i)
        x_data.append(i)
        model_output = run_model(vfromday=i,use_priority=False)[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 1st sim",model_output[-20:])
            run_model(vfromday=i, use_priority=False, print_model=True)[3].sum(axis=0)
        deaths_data.append(model_output[-1])
        model_output = run_model(vfromday=i,use_priority=True)[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 2nd sim",model_output[-20:])
            run_model(vfromday=i, use_priority=True, print_model=True)[3].sum(axis=0)
        deaths_data2.append(model_output[-1])
        model_output = run_model(vfromday=i,use_priority=True,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17])[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 3rd sim",model_output[-20:])
            run_model(vfromday=i, use_priority=True,
                      priority=[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17], print_model=True)[3].sum(axis=0)
        deaths_data3.append(model_output[-1])
    # plt.title("Effect of daily vaccination capacity on total number of deaths")
    xy_dict = {'Vaccination delay (in days)': x_data, 'Total number of deaths': deaths_data}
    df_deaths_data = pd.DataFrame(xy_dict)
    xy_dict_all3 = {'Vaccination delay (in days)': x_data+x_data+x_data, 'Total number of deaths': deaths_data+deaths_data2+deaths_data3, 'series': ['Uniform strategy' for i in range(len(x_data))]+['Age-based strategy' for i in range(len(x_data))]+['Contact-based strategy' for i in range(len(x_data))]}
    df_deaths_data_all3 = pd.DataFrame(data=xy_dict_all3)
    # shading(sns.lineplot(x='Vaccination delay (in days)', y='Total number of deaths', data=df_deaths_data,dashes=False,linewidth=4),1,0.1,[palette_five[3]],legend=False,xlim=False)
    shading(sns.lineplot(x='Vaccination delay (in days)', y='Total number of deaths', hue='series', data=df_deaths_data_all3, dashes=False,linewidth=2), 3, 0.1, [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.3126890019504329, 0.6928754610296064, 0.1923704830330379), (0.23299120924703914, 0.639586552066035, 0.9260706093977744)], legend=False, xlim=False)
    ax = plt.gca()
    # print(ax.get_lines()[-3].get_c(color_format='hex'))
    # print(ax.get_lines()[-2].get_c(color_format='hex'))
    # print(ax.get_lines()[-1].get_c(color_format='hex'))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlim(right=x_data[-1])
    ax.tick_params(axis="y",direction="out", pad=-3)
    ax.tick_params(axis="x",direction="out", pad=-2)
    plt.savefig('graphics/vax-r.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

#  vax per day
if make_vax_per_day:
    grouping_key = ["Under 20","20--39","40--59","60--79","80+"]
    # plt.figure(figsize=(4,2.5)) # for poster
    sns.set_palette(palette_five)
    x_data = []
    deaths_datal = []
    for i in range(151):
        if i%10 == 0:
            print(i)
        x_data.append(i*2000)
        deaths_datal.append(run_model(vperday=i*2000,use_priority=False)[3][:,-1])
    deaths_data=np.array(deaths_datal)
    print(deaths_data[:,0])
    grouped_deaths = np.zeros((5,len(deaths_data[:,0])))
    # grouped_deaths[0] = np.sum(deaths_data[:,:5],axis=1)
    for i in range(0,4):
        grouped_deaths[i] = np.sum(deaths_data[:,4*i:4*i+4],axis=1)
    grouped_deaths[4] = np.sum(deaths_data[:,16:],axis=1)
    plt.figure(figsize=set_size(textwidth))
    plt.xlabel("Initial vaccination capacity (IVC)")
    plt.ylabel("Total number of deaths")
    # plt.title("Total deaths for varying daily vaccination capacity (oldest first)") # for pres
    df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
    plt.stackplot([2000*i for i in range(len(grouped_deaths[0]))],grouped_deaths[0],grouped_deaths[1],grouped_deaths[2],grouped_deaths[3],grouped_deaths[4], edgecolor="none", alpha=0.9)

    plt.xlim(right=2000*(len(deaths_data[:,0])-1))
    plt.legend(grouping_key, loc="upper right")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x",which="major",bottom=True,color="lightgray")
    plt.savefig('graphics/vax-capacity-stacked-noprio-poster.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()

if make_vax_per_day_daily:
    imult = 10000
    grouping_key = ["Under 20","20--39","40--59","60--79","80+"]
    # plt.figure(figsize=(4,2.5)) # for poster
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3,1)))
    # plt.title("Total deaths with varying vaccination start dates (no priority)") # for pres
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],[3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(102):
            if i%10 == 0:
                print(i)
            x_data.append(i*imult)
            deaths_datal.append(run_model(vperday=i*imult, vaxcapacityincrease=i*imult/100, vaxmax=i*imult*5, use_priority=useprio_key[graphnum],priority=prio_key[graphnum])[3][:,-1])
        deaths_data=np.array(deaths_datal)
        # print(deaths_data[:,0])
        grouped_deaths = np.zeros((5,len(deaths_data[:,0])))
        # grouped_deaths[0] = np.sum(deaths_data[:,:5],axis=1)
        for i in range(0,4):
            grouped_deaths[i] = np.sum(deaths_data[:,4*i:4*i+4],axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:,16:],axis=1)
        # print(grouped_deaths)
        lives_saved = np.zeros(len(deaths_data[:,0]))
        novaxdeaths = sum(run_model(vperday=0)[3][:,-1])
        for i in range(len(deaths_data[:,0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:,i])
            # print(i, sum(grouped_deaths[:,i]))
        # print("lives saved:",lives_saved)
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([imult*i for i in range(len(grouped_deaths[0])-1)],cumulative_to_daily(grouped_deaths[0],absol=True),cumulative_to_daily(grouped_deaths[1],absol=True),cumulative_to_daily(grouped_deaths[2],absol=True),cumulative_to_daily(grouped_deaths[3],absol=True),cumulative_to_daily(grouped_deaths[4],absol=True), edgecolor="none", alpha=0.9)
        # for num, i in enumerate(np.array(cumulative_to_daily(grouped_deaths[0]))+np.array(cumulative_to_daily(grouped_deaths[1]))+np.array(cumulative_to_daily(grouped_deaths[2]))+np.array(cumulative_to_daily(grouped_deaths[3]))+np.array(cumulative_to_daily(grouped_deaths[4]))):
        # for num, i in enumerate(np.array(cumulative_to_daily(grouped_deaths[4]))):
        #     print(num+200, i)
        # ax[graphnum].set_xlim(right=imult*len(deaths_data[:,0]-1))
        ax[graphnum].set_ylim(top=7000)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="upper right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Initial vaccination capacity (IVC)")
        # ax[graphnum].plot([imult*i for i in range(len(grouped_deaths[0]))], lives_saved, '--', color=palette_neutral[1])
        # ax = plt.gca()
        if graphnum == 1:
            ax[graphnum].set_ylabel("Number of lives saved by increasing initial vaccination capacity by 10,000")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[graphnum].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x",which="major",bottom=True,color="lightgray",width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-capacity-stacked-age-prio3-daily.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()

# vax start date (cumulative)
if make_vax_per_day_cum:
    imult = 20000
    grouping_key = ["Under 20", "20–39", "40–59", "60–79", "80+"]
    # plt.figure(figsize=(4,2.5)) # for poster
    sns.set_palette(palette_five)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(set_size(textwidth, fraction=0.7, subplots=(3, 1)))
    # plt.title("Total deaths with varying vaccination start dates (no priority)") # for pres
    useprio_key = [False, True, True]
    prio_key = [[17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [3, 2, 1, 8, 4, 7, 9, 5, 6, 10, 0, 11, 12, 13, 14, 15, 16, 17]]
    title_key = ["(a) Uniform strategy.", "(b) Age-based strategy.", "(c) Contact-based strategy."]
    for graphnum in range(3):
        x_data = []
        deaths_datal = []
        for i in range(101):
            if i % 10 == 0:
                print(i*imult)
            x_data.append(i)
            deaths_datal.append(
                run_model(vperday=i*imult, vaxcapacityincrease=i*imult/100, vaxmax=i*imult*5, use_priority=useprio_key[graphnum], priority=prio_key[graphnum])[3][:, -1])
        deaths_data = np.array(deaths_datal)
        # print(deaths_data[:,0])
        grouped_deaths = np.zeros((5, len(deaths_data[:, 0])))
        # grouped_deaths[0] = np.sum(deaths_data[:,:5],axis=1)
        for i in range(0, 4):
            grouped_deaths[i] = np.sum(deaths_data[:, 4 * i:4 * i + 4], axis=1)
        grouped_deaths[4] = np.sum(deaths_data[:, 16:], axis=1)
        # print(grouped_deaths)
        lives_saved = np.zeros(len(deaths_data[:, 0]))
        novaxdeaths = sum(run_model(vperday=0)[3][:, -1])
        for i in range(len(deaths_data[:, 0])):
            lives_saved[i] = novaxdeaths - sum(grouped_deaths[:, i])
        # print("lives saved:",lives_saved)
        df_deaths = pd.DataFrame(np.column_stack([grouped_deaths[i] for i in range(5)]), columns=grouping_key)
        ax[graphnum].title.set_text(title_key[graphnum])
        ax[graphnum].stackplot([imult*i for i in range(len(grouped_deaths[0]))], grouped_deaths[0], grouped_deaths[1],
                               grouped_deaths[2], grouped_deaths[3], grouped_deaths[4], edgecolor="none", alpha=0.9)
        ax[graphnum].set_xlim(right=imult*len(deaths_data[:, 0]) - 1)
        if graphnum == 0:
            ax[graphnum].legend(grouping_key, loc="center right")
        if graphnum == 2:
            ax[graphnum].set_xlabel("Initial vaccination capacity (IVC)")
        ax[graphnum].plot([imult*i for i in range(len(grouped_deaths[0]))], lives_saved, 'k--')
        # print(lives_saved)
        # ax = plt.gca()
        if graphnum == 1:
            ax[graphnum].set_ylabel("Total number of deaths")
        ax[graphnum].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[graphnum].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        for spine in ax[graphnum].spines.values():
            spine.set_visible(False)
        ax[graphnum].tick_params(axis="x", which="major", bottom=True, color="lightgray", width=1)
        ax[graphnum].tick_params(axis="y", direction="out", pad=-4)
    fig.tight_layout()
    fig.savefig('graphics/vax-capacity-stacked-age-prio3-cum.pdf', dpi=400, bbox_inches="tight", transparent=True)
    fig.show()






# vax per day
if make_vax_per_day_samegraph:
    imult = 20000
    plt.figure(figsize=set_size(textwidth))
    sns.set_palette([palette_five[3]])
    x_data = []
    deaths_data = []
    deaths_data2 = []
    deaths_data3 = []
    for i in range(41):
        if i % 10 == 0:
            print(i,i*imult)
        x_data.append(i*imult)
        model_output = run_model(vperday=i*imult,vfromday=200,use_priority=False,vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 1st sim",model_output[-20:])
            run_model(vperday=i*imult,vfromday=200,use_priority=False,print_model=True,vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        deaths_data.append(model_output[-1])
        model_output = run_model(vperday=i*imult,vfromday=200,use_priority=True,vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 2nd sim",model_output[-20:])
            run_model(vperday=i*imult,vfromday=200,use_priority=True,print_model=True,vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        deaths_data2.append(model_output[-1])
        model_output = run_model(vperday=i*imult,vfromday=200,use_priority=True,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17],vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        if abs(model_output[-1] - model_output[-10]) > 0.01:
            print("ERROR ON",i,"in 3rd sim",model_output[-20:])
            run_model(vperday=i*imult,vfromday=200,use_priority=True,priority=[3,2,1,8,4,7,9,5,6,10,0,11,12,13,14,15,16,17],print_model=True,vaxcapacityincrease=i*imult/100,vaxmax=i*5*imult)[3].sum(axis=0)
        deaths_data3.append(model_output[-1])
    # plt.title("Effect of daily vaccination capacity on total number of deaths")
    xy_dict = {'Initial vaccination capacity (IVC)': x_data, 'Total number of deaths': deaths_data}
    df_deaths_data = pd.DataFrame(xy_dict)
    xy_dict_all3 = {'Initial vaccination capacity (IVC)': x_data+x_data+x_data, 'Total number of deaths': deaths_data+deaths_data2+deaths_data3, 'series': ['Uniform strategy' for i in range(len(x_data))]+['Age-based strategy' for i in range(len(x_data))]+['Contact-based strategy' for i in range(len(x_data))]}
    df_deaths_data_all3 = pd.DataFrame(data=xy_dict_all3)
    # df_deaths_data_all3 = df_deaths_data_all3.set_index([["noprio", "oldestprio", "contactprio"]])
    # shading(sns.lineplot(x='Initial vaccination capacity (IVC)', y='Total number of deaths', data=df_deaths_data,dashes=False,linewidth=4),1,0.1,[palette_five[3]],legend=False,xlim=False)
    shading(sns.lineplot(x='Initial vaccination capacity (IVC)', y='Total number of deaths', hue='series', data=df_deaths_data_all3, dashes=False,linewidth=2), 3, 0.1, [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.3126890019504329, 0.6928754610296064, 0.1923704830330379), (0.23299120924703914, 0.639586552066035, 0.9260706093977744)], legend=False, xlim=False)
    ax = plt.gca()
    # print(ax.get_lines()[-3].get_c(color_format='hex'))
    # print(ax.get_lines()[-2].get_c(color_format='hex'))
    # print(ax.get_lines()[-1].get_c(color_format='hex'))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlim(right=x_data[-1])
    ax.tick_params(axis="y",direction="out", pad=-3)
    ax.tick_params(axis="x",direction="out", pad=-2)
    plt.savefig('graphics/vax-capacity.pdf', dpi=400, bbox_inches="tight", transparent=True)
    plt.show()




