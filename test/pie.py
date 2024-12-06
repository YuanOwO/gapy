import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

font = fm.FontEntry(fname='../.fonts/Noto_Sans_TC/NotoSansTC-Regular.ttf', name='Noto Sans TC')
fm.fontManager.ttflist.append(font)

mpl.rcParams["font.family"] = font.name
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 'xx-large'


fig = plt.figure(figsize=(5, 7), dpi=200)
ax = fig.add_subplot()
ax.set_title('使用輪盤法的選擇父代', pad=20)

columns = ('A', 'B', 'C', 'D', 'E')
rows = ('距離', '適應度', '機率')

distance = np.array([10, 12, 18, 25, 30])

fitness = 1 / distance
p = fitness / np.sum(fitness)


text = [
    [round(d, 3) for d in distance],
    [round(f, 3) for f in fitness],
    [f'{p * 100:1.1f}%' for p in p]
]

colors = plt.get_cmap('binary')(np.linspace(0, 1, len(columns)))
colors = None

ax.pie(p, colors=colors, labels=columns,
       radius=3.5, center=(4, 4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"},
       #    frame=False,
       autopct='%1.1f%%',
       counterclock=False, startangle=-270,
       )

ax.set(xlim=(0, 8), xticks=[], ylim=(0, 8), yticks=[])

tab = ax.table(
    cellText=text,
    rowLabels=rows,
    colLabels=columns,
    loc='bottom', cellLoc='center', rowLoc='center',)

tab.scale(xscale=1, yscale=2)

fig.savefig('pie.png')
