
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rc('font',family='Times New Roman')

color = {}
# color['green'] = (165/255.,190/255.,106/255.)
# color['blue'] = (6/255.,156/255.,207/255.)
# color['red'] = (153/255.,51/255.,0/255.)
# color['yellow'] = (248/255.,250/255.,13/255.)
color['green'] = (97/255.,144/255.,136/255.)
color['blue'] = (143/255.,170/255.,220/255.)
color['red'] = (153/255.,51/255.,0/255.)
color['yellow'] = (222/255.,149/255.,38/255.)

# predicates = ["behind", "near", "under", "holding", "watching", "wearing"]
# imp = [11.06, 06.87, 01.83, 11.26, 00.00, 19.28]
# imp_kgc = [14.25, 16.30, 07.31, 16.38, 01.43, 19.67]
# motifs = [12.67, 10.75, 02.33, 07.46, 00.00, 05.66]
# motifs_kgc = [15.88, 15.50, 07.14, 11.20, 03.33, 09.30]
# vctree = [12.67, 10.75, 02.33, 07.46, 00.00, 05.66]
# vctree_kgc = [18.24, 15.06, 10.72, 15.89, 03.21, 08.57]
# tde = [03.76, 02.75, 03.49, 03.35, 00.00, 04.88]
# tde_kgc = [12.24, 23.93, 22.65, 26.28, 10.17, 05.23]

predicates = ["behind", "holding", "near", "under", "in"]
imp = [01.65, 00.63, 03.53, 00.66, 02.13]
tde = [05.03, 01.26, 00.16, 03.16, 00.19]
motifs = [01.41, 01.26, 03.34, 00.00, 01.74]
motifs_we = [01.76, 00.63, 03.83, 00.66, 02.41]
motifs_rgn = [06.84, 03.77, 05.62, 05.87, 03.91]
vctree = [04.04, 01.26, 04.80, 02.49, 00.71]
vctree_we = [04.31, 01.26, 05.00, 02.44, 01.54]
vctree_rgn = [06.47, 04.40, 07.16, 03.49, 04.01]

# predicates = ["above", "behind", "holding", "in", "near", "under"]
# imp = [00.68, 01.65, 00.63, 02.13, 03.53, 00.66]
# tde = [08.87, 05.03, 01.26, 00.19, 00.16, 03.16]
# motifs = [00.68, 01.41, 01.26, 01.74, 03.34, 00.00]
# motifs_we = [00.34, 01.76, 00.63, 02.41, 03.83, 00.66]
# motifs_rgn = [05.46, 06.84, 03.77, 03.91, 05.62, 05.87]
# vctree = [07.68, 04.04, 01.26, 00.71, 04.80, 02.49]
# vctree_we = [04.10, 04.31, 01.26, 01.54, 05.00, 02.44]
# vctree_rgn = [05.52, 06.47, 04.40, 04.01, 07.16, 03.49]


xticks = np.arange(len(predicates))
sns.set_theme(style="whitegrid")
col_width = 0.10
split = 0.02
half_col_width = (col_width/2)*1.05
half_split = split/2
fig, ax = plt.subplots(figsize=(15, 7))


# ax.bar(xticks - 2*half_col_width, vctree, width=col_width, label="VCT", color=color['green'], hatch="xxx")
#
# ax.bar(xticks, vctree_we, width=col_width, label="VCT-WE", color=color['yellow'], hatch="///")
#
# ax.bar(xticks + 2*half_col_width, vctree_rgn, width=col_width, label="VCT-RGN", color=color['red'],hatch="")
#
ax.bar(xticks - 7*half_col_width, imp, width=col_width, label="IMP", hatch="xxxx",color=color['yellow'])

ax.bar(xticks - 5*half_col_width, tde, width=col_width, label="TDE", hatch="xxxx",color=color['blue'])

ax.bar(xticks - 3*half_col_width, motifs, width=col_width, label="MOTIFS", hatch="xxxx",color=color['green'])

ax.bar(xticks - 1*half_col_width, motifs_we, width=col_width, label="MOTIFS-WE", hatch="///",color=color['green'])

ax.bar(xticks + 1*half_col_width, motifs_rgn, width=col_width, label="MOTIFS-RGN",color=color['green'])

ax.bar(xticks + 3*half_col_width, vctree, width=col_width, label="VCT", hatch="xxxx",color=color['red'])

ax.bar(xticks + 5*half_col_width, vctree_we, width=col_width, label="VCT-WE", hatch="///",color=color['red'])

ax.bar(xticks + 7*half_col_width, vctree_rgn, width=col_width, label="VCT_RGN",color=color['red'])


FONT_SIZE = 20
LEGEND_FONT_SIZE = 15
# ax.set_title("Grouped Bar plot", fontsize=15)
# ax.set_xlabel("Predicates", fontsize=FONT_SIZE)
ax.set_ylabel("mR@20 (in percentage)", fontsize=FONT_SIZE)

ax.legend(fontsize=LEGEND_FONT_SIZE)

# 最后调整x轴标签的位置
ax.set_xticks(xticks)
ax.set_xticklabels(predicates,fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)

plt.savefig(f'./mZero@20.pdf', bbox_inches='tight')
plt.show()















#
## data1=[['aaa',1],['bbb',2]]
# data2=[['aaa',2],['bbb',4]]
# ax=sns.barplot(x='method', y='value', data=df)
#
#
# ax.set_ylabel('value')
# ax.set_ylim([0,8])
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(14)
#


# for bar in ax.patches:
#     bar.set_width(2.6)
#     x = bar.get_x()
#     width = bar.get_width()
#     centre = x + width / 2.
#     bar.set_x(centre-2.2)

# plt.savefig('./test.png')
# df = pd.DataFrame()
# df['DGFed']=[1]
# df[f'EFL (G={N//10})'] = [efl(N=N,G=N//10) for _ in range(25)]
# df[f'EFL (G={N//20})'] = [efl(N=N,G=N//10) for _ in range(25)]