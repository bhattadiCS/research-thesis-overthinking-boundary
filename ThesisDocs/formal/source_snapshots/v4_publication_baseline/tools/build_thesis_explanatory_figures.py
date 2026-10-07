"""Render explanatory diagrams without changing any scientific observation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ThesisDocs/images/thesis_v4"

def box(ax, center, size, text, *, color="#EAF3F8"):
    x,y=center; w,h=size
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.008,rounding_size=0.012",facecolor=color,
        edgecolor="#33566E",linewidth=1.2))
    ax.text(x,y,text,ha="center",va="center",fontsize=13,color="#111111",linespacing=1.4)

def arrow(ax, a,b,label=None,offset=(0,0)):
    ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":1.5,"color":"#33566E"})
    if label:
        ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,
                ha="center",va="center",fontsize=13,color="#111111")

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"savefig.dpi":240})
    fig,ax=plt.subplots(figsize=(7.6,4.9));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    for x,text in [(.16,"Archived evidence\nVersioned grading"),
                   (.5,"Task partitions\nFit and calibrate"),
                   (.84,"Frozen policy\nFeature contract")]:
        box(ax,(x,.83),(.285,.18),text)
    arrow(ax,(.31,.83),(.35,.83));arrow(ax,(.65,.83),(.69,.83))
    box(ax,(.16,.43),(.285,.22),"Observed prefix\nCurrent and past\nobservations only")
    box(ax,(.5,.43),(.285,.22),"Decision rule\nResponse floor\nand horizon guards")
    box(ax,(.84,.43),(.285,.22),"Stop: emit answer\nContinue: request\nthe next response")
    arrow(ax,(.31,.43),(.35,.43));arrow(ax,(.65,.43),(.69,.43))
    arrow(ax,(.84,.73),(.84,.66));arrow(ax,(.84,.66),(.5,.66));arrow(ax,(.5,.66),(.5,.55))
    ax.text(.48,.70,"Runtime uses frozen parameters",ha="center",fontsize=13)
    arrow(ax,(.84,.31),(.84,.21));arrow(ax,(.84,.21),(.16,.21));arrow(ax,(.16,.21),(.16,.31))
    ax.text(.5,.245,"Continue adds a response and incurred cost",ha="center",fontsize=13)
    ax.text(.5,.065,"Reference labels enter fitting and evaluation offline.\nThey are excluded from runtime decision inputs.",
            ha="center",va="center",fontsize=13,linespacing=1.4)
    fig.savefig(OUT/"information_flow.png");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.6,4.6));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,(.16,.72),(.27,.22),"t = 0\nCorrectness = 0\nCost so far = 0.00")
    box(ax,(.5,.72),(.27,.22),"t = 1\nCorrectness = 0\nCost so far = 0.10")
    box(ax,(.84,.72),(.27,.22),"t = 2\nCorrectness = 1\nCost so far = 0.20")
    arrow(ax,(.3,.72),(.35,.72));arrow(ax,(.64,.72),(.69,.72))
    ax.text(.33,.93,"Continue\nCost 0.10",ha="center",va="center",fontsize=13,linespacing=1.3)
    ax.text(.67,.93,"Continue\nCost 0.10",ha="center",va="center",fontsize=13,linespacing=1.3)
    for x,text,color in [(.16,"Stop\nReward = 0.00","#F8ECE5"),
                         (.5,"Stop\nReward = -0.10","#F8ECE5"),
                         (.84,"Forced stop\nReward = 0.80","#E8F4EC")]:
        box(ax,(x,.33),(.27,.18),text,color=color);arrow(ax,(x,.6),(x,.43))
    ax.text(.5,.095,"Immediate drift at t = 0 is -0.10.\nContinuing to the horizon gives reward 0.80.",
            ha="center",va="center",fontsize=13,linespacing=1.4)
    fig.savefig(OUT/"delayed_repair_tree.png");plt.close(fig)
    print(str(OUT))

if __name__=="__main__":build()
