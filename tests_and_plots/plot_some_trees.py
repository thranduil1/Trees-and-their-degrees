## Some examples of trees plotted for different values. 

family = make_tree_family(
    a=3,
    b=2,
    N=3,
)

x = np.array([0.8, 0.8, 0.8])

times = [0.0, 0.5, 1.0]

titles = [
    r"$t=0$: columns grouped first",
    r"$t=1/2$: one ab-junction",
    r"$t=1$: rows grouped first",
]

fig = plt.figure(figsize=(18, 6))

for k, (t, title) in enumerate(
    zip(times, titles),
    start=1,
):
    ax = fig.add_subplot(
        1,
        3,
        k,
        projection="3d",
    )

    tree = tree_at(
        family,
        x=x,
        t=t,
    )

    plot_tree_3d(
        tree,
        ax=ax,
        title=title,
        show_labels=True,
    )

plt.tight_layout()
plt.show()
