import matplotlib.pyplot as plt


def pie_chart(x, label, color, is_explode: bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    explode = (0.1, 0, 0)
    if is_explode:
        ax.pie(x=x, labels=label, explode=explode, colors=color,
               autopct='%1.1f%%',
               textprops={'fontsize': 11})
    else:
        ax.pie(x=x, labels=label, colors=color,
               autopct='%1.1f%%',
               textprops={'fontsize': 11})
    ax.set_title('sentimen terhadap sistem tilang elektronik di twitter', fontsize=12)
    return fig, ax


def bar_chart(x, height, sizes, color):
    fig, ax = plt.subplots()
    ax.bar(x, height, color=color)
    for i in range(len(height)):
        ax.text(i, sizes[i], sizes[i], ha='center')
    ax.set_xlabel('Nilai Prediksi')
    ax.set_ylabel('Jumlah')
    ax.set_title('Diagram Batang Prediksi')
    return fig, ax
