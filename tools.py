from adjustText import adjust_text
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def connect_cursors(mpl_obj, labels):
    if isinstance(labels, int):
        labels = [f"e{i}" for i in range(labels)]
    @mpl_obj.connect("add")
    def _(sel):
        xy_data = sel.target
        sel.annotation.set_text(f"Entry: {labels[sel.index]}\nIndex: {sel.index}\n(X,Y): ({xy_data[0]:.1f},{xy_data[1]:.1f})")
        sel.annotation.set(fontsize=8)
        sel.annotation.get_bbox_patch().set(fc="skyblue", alpha=0.75, boxstyle="square")
        sel.annotation.arrow_patch.set(arrowstyle="-")

def add_model_statistics(ax,statistics, loc='upper left'):
    strings_statistics = []
    for k,v in statistics.items():
        strings_statistics.append("{:<15}{:>6.3f}".format(k,v))

    metrics_box = AnchoredText(
        "\n".join(strings_statistics),
        loc=loc,
        pad=1,
        frameon=False,
        prop={
            'horizontalalignment':'left',
            'fontsize':8,
            'fontfamily':'monospace',
            'bbox':dict(facecolor='white',alpha=0.5)
        }
    )
    ax.add_artist(metrics_box)

def add_labels(x, y, labels, ax):
    texts = []
    for xi,yi,label in zip(x,y,labels):
        texts.append(ax.text(xi,yi,label,fontsize=8))   
    adjust_text(texts, ax=ax)