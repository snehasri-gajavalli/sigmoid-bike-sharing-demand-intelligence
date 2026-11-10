"""
Generate a clean architecture diagram for the current Streamlit prototype.

Place this file in the project root and run:
    python generate_prototype_architecture.py

Output:
    outputs/prototype_architecture.png
    outputs/prototype_screenshots/  (save genuine app screenshots here)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
SCREENSHOTS = OUT_DIR / "prototype_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "prototype_architecture.png"

fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 17)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

NAVY, DARK, MUTED = "#17324D", "#263746", "#637586"
BLUE, TEAL, GOLD, PURPLE, PALE = "#DCEBFA", "#DDF1EE", "#FFF0D8", "#E9E3F5", "#EDF2F6"
EDGE = "#91A5B6"

def node(x, y, w, h, title, detail="", fill="white", title_size=11, detail_size=8.8):
    rect = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor=fill,edgecolor=EDGE,linewidth=1.35)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h*.66,title,ha="center",va="center",fontsize=title_size,
            color=NAVY,fontweight="bold")
    if detail:
        ax.text(x+w/2,y+h*.31,detail,ha="center",va="center",fontsize=detail_size,
                color=DARK,linespacing=1.25)
    return (x,y,w,h)

def arr(a,b,color=MUTED,style="-",lw=1.6):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,
                                 linewidth=lw,color=color,linestyle=style))

ax.text(8.5,9.65,"BIKE SHARING DEMAND INTELLIGENCE",ha="center",va="center",
        fontsize=19,fontweight="bold",color=NAVY)
ax.text(8.5,9.25,"Natural-language prototype architecture | Current Streamlit implementation",
        ha="center",va="center",fontsize=10.5,color=MUTED)

# Main assistant input flow
node(.45,7.75,2.55,1.05,"Operations manager","Types a question in plain English",BLUE)
node(3.75,7.75,3.0,1.05,"Streamlit Business Assistant","Question text box + Analyze Question",TEAL)
node(7.55,7.75,3.1,1.05,"Question normalisation","Lowercase + trim whitespace",GOLD)
node(11.45,7.75,4.95,1.05,"Keyword-based intent router","Ordered if / elif keyword checks",PURPLE)
arr((3.0,8.275),(3.75,8.275))
arr((6.75,8.275),(7.55,8.275))
arr((10.65,8.275),(11.45,8.275))

# Data source banner - loaded once and shared by the handlers
node(4.45,6.25,8.1,.72,"Shared historical data","data/hour.csv is loaded at startup as hour_df and used by the handlers",PALE,
     title_size=10.5,detail_size=8.5)

# Branch router into 5 intent handlers
branch_xs=[.35,3.72,7.09,10.46,13.83]
branch_titles=[
    ("Weather","get_weather_summary()\nweather-group statistics"),
    ("Customer segments","Pandas groupby\ncasual / registered shares"),
    ("Day-type comparison","Pandas groupby\nmean cnt by workingday"),
    ("Peak demand","get_hourly_profile()\nbusiest hour and top hours"),
    ("Fleet / allocation","demand_timing_recommendations()\ntiming + pilot guidance"),
]
# Connect router down to a horizontal distribution line
arr((13.9,7.75),(13.9,5.55),color="#8195A5")
ax.plot([1.85,15.35],[5.55,5.55],color="#8195A5",lw=1.5)
for x,(title,detail) in zip(branch_xs,branch_titles):
    cx=x+1.42
    arr((cx,5.55),(cx,5.05),color="#8195A5")
    node(x,3.85,2.85,1.18,title,detail,fill="white",title_size=10,detail_size=8.2)

# Outputs / unmatched question
node(2.15,2.05,7.15,1.0,"Streamlit answer rendering",
     "Metrics, tables, explanatory text and limitation warnings",TEAL)
node(11.0,2.05,4.95,1.0,"Unmatched question",
     "Shows example topics the assistant can answer",GOLD)
# Branch group arrows down to outputs
ax.plot([1.77,15.25],[3.45,3.45],color="#8195A5",lw=1.4,alpha=.9)
for x,_ in zip(branch_xs,branch_titles):
    cx=x+1.42
    arr((cx,3.85),(cx,3.45),color="#8195A5")
# feed each group to its result container
arr((5.7,3.45),(5.7,3.05),color="#8195A5")
arr((13.45,3.45),(13.45,3.05),color="#A47B38",style="--")

# Independent model prediction workflow
ax.text(8.5,1.55,"SEPARATE MODEL-BASED PREDICTION WORKFLOW",
        ha="center",va="center",fontsize=9.5,fontweight="bold",color=MUTED)
node(.65,.35,4.0,.85,"Demand Prediction form",
     "Date, hour and weather / temperature inputs",BLUE,title_size=10,detail_size=8)
node(6.15,.35,4.45,.85,"Feature building + inference",
     "build_prediction_row() → make_model_matrix() → predict_demand() → display result",PURPLE,
     title_size=10,detail_size=7.8)
node(12.15,.35,4.15,.85,"Saved model bundle",
     "models/bike_demand_model_bundle.pkl",PALE,title_size=10,detail_size=8)
arr((4.65,.78),(6.15,.78))
arr((12.15,.78),(10.6,.78))
ax.text(8.5,.08,
        "The Business Assistant is rule-based; the saved model bundle is used by the separate Demand Prediction page.",
        ha="center",va="center",fontsize=8,color="#914D2E",fontweight="bold")

fig.savefig(OUT,dpi=220,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Architecture diagram saved: {OUT}")
print(f"Save actual Streamlit screenshots in: {SCREENSHOTS}")
