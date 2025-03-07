import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random as r
import numpy as np
from array import array
from scipy import stats as st

times = 100000
m0 = 2
m = 2


# ======================================================================================================================

def remove_all(list, val):
    return [value for value in list if value != val]


def n_connected(n):
    problist = []
    G = nx.Graph()
    for node in range(1, n + 1):
        G.add_node(node)
        problist.append(node)

    for node1 in problist:
        for node2 in problist:
            if node1 != node2:
                G.add_edge(node1, node2)
    return G, problist


# ======================================================================================================================

G_rand, problist_rand = n_connected(m0)

# print("pre.", problist)
for t in range(times):
    G_rand.add_node(t + m0 + 1)
    tmplist = problist_rand.copy()
    for i in range(m):
        rand_node = r.randint(1, len(problist_rand))
        G_rand.add_edge(rand_node, t + m0 + 1)
        # print(rand_node)

    problist_rand.append(t + m0 + 1)
    # print(t, ". ", problist)

# ======================================================================================================================

G_BA, problist_BA = n_connected(m0)

for node in range(1, m0 + 1):
    G_BA.add_node(node)
    problist_BA.append(node)

for node1 in problist_BA:
    for node2 in problist_BA:
        if node1 != node2:
            G_BA.add_edge(node1, node2)

# print("pre.", problist)
for t in range(times):
    G_BA.add_node(t + m0 + 1)
    tmplist = problist_BA.copy()
    for i in range(m):
        rand_index = r.randint(0, len(tmplist) - 1)
        rand_node = tmplist[rand_index]

        # print("  list[", rand_index, "] = ", rand_node)
        # tmplist.remove(rand_node)
        tmplist = remove_all(tmplist, rand_node)
        # print(" ", tmplist)
        G_BA.add_edge(rand_node, t + m0 + 1)
        problist_BA.append(rand_node)

    problist_BA.append(t + m0 + 1)
    # print(t, ". ", problist)

# ======================================================================================================================

hist_rand = nx.degree_histogram(G_rand)
hist_rand = np.array(hist_rand, dtype=float)
hist_rand = hist_rand / times
# print(len(hist_rand))
# print(hist_rand)

x_rand = np.arange(3, len(hist_rand) + 1, )
y_rand = np.log10(hist_rand[2:9999])
# a_rand, b_rand, r, p, std, = st.linregress(x_rand, y_rand)
# print("a = ", a_rand)

fig1 = plt.Figure()
plt.scatter(x_rand, y_rand, c='k', s=12)
# plt.plot(x_rand, a_rand * x_rand + b_rand, 'g--')
plt.ylabel("log(N)")
plt.xlabel(r"$k_i$")
# plt.title("regress: a = " + str(a_rand) + ", b = " + str(b_rand))
plt.suptitle("Random Network for t = " + str(times))
plt.show()

# ======================================================================================================================

hist_BA = nx.degree_histogram(G_BA)
remove_all(hist_BA, 0)
hist_BA = np.array(hist_BA, dtype=float)
hist_BA = hist_BA / times
# print(len(hist_rand))
# print(hist_rand)

fig2 = plt.Figure()
plt.scatter(np.log10(np.arange(3, len(hist_BA) + 1)), np.log10(hist_BA[2:9999]), c='k', s=12)
plt.ylabel("log(N)")
plt.xlabel(r"log($k_i$)")
plt.suptitle("Barabasi-Albert Network for t = " + str(times))
plt.show()

# nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
# plt.show()
