import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import networkx as nx
import matplotlib.pyplot as plt

nltk.download("stopwords")


def related_words(fake_path, true_path):
    # Set up empty lists to hold the text contents of each file
    fake_news_text = []
    true_news_text = []

    # Loop through all files in the Fake folder and add their contents to the fake_news_text list
    for filename in os.listdir(fake_path):
        if filename.endswith(".txt"):
            with open(os.path.join(fake_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                fake_news_text.append(text)

    # Loop through all files in the True folder and add their contents to the true_news_text list
    for filename in os.listdir(true_path):
        if filename.endswith(".txt"):
            with open(os.path.join(true_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                true_news_text.append(text)

    # Tokenize the text into words and remove punctuation and stop words
    stop_words = set(stopwords.words("english"))
    fake_news_words = []
    true_news_words = []

    for text in fake_news_text:
        tokens = nltk.word_tokenize(text.lower())
        fake_news_words.extend(
            [word for word in tokens if word.isalpha() and word not in stop_words])

    for text in true_news_text:
        tokens = nltk.word_tokenize(text.lower())
        true_news_words.extend(
            [word for word in tokens if word.isalpha() and word not in stop_words])

    # Extract the most common collocations in each case
    bigram_measures = BigramAssocMeasures()
    fake_news_finder = BigramCollocationFinder.from_words(fake_news_words)
    true_news_finder = BigramCollocationFinder.from_words(true_news_words)

    fake_news_collocations = fake_news_finder.nbest(bigram_measures.pmi, 20)
    true_news_collocations = true_news_finder.nbest(bigram_measures.pmi, 20)

    # Create a graph of related words in both cases
    G = nx.Graph()

    for word1, word2 in fake_news_collocations:
        if word1 not in G:
            G.add_node(word1, color="red")
        if word2 not in G:
            G.add_node(word2, color="red")
        G.add_edge(word1, word2, weight=1, color="red")

    for word1, word2 in true_news_collocations:
        if word1 not in G:
            G.add_node(word1, color="blue")
        if word2 not in G:
            G.add_node(word2, color="blue")
        G.add_edge(word1, word2, weight=1, color="blue")

    # Find the longest chain of interconnected words in fake news articles and color them yellow
    fake_news_chains = list(nx.connected_components(G.subgraph(
        [word for word, color in nx.get_node_attributes(G, "color").items() if color == "red"])))
    fake_news_longest_chain = max(fake_news_chains, key=len)
    node_colors = ["yellow" if node in fake_news_longest_chain else color for node,
                   color in nx.get_node_attributes(G, "color").items()]

    # Draw the graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Find the longest chain of interconnected words in fake news articles
    fake_chains = []
    for node in G.nodes():
        if G.nodes[node]["color"] == "red":
            chain = []
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]["color"] == "red":
                    chain.append(node)
                    chain.append(neighbor)
                    fake_chains.append(chain)

    longest_chain = max(fake_chains, key=len)
    print(
        f"The longest chain of interconnected words in fake news articles is: {longest_chain}")

    # Color the nodes in the longest chain yellow
    node_colors = []
    for node in G.nodes():
        if node in longest_chain:
            node_colors.append("yellow")
        else:
            node_colors.append(G.nodes[node]["color"])

    # Draw the graph with colored nodes
    edge_colors = [edge[2]["color"] for edge in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=12)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [(weight - min(edge_weights) + 1) * 3 for weight in edge_weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color=edge_colors, alpha=0.7)

    # Set the title and display the graph
    plt.axis("off")
    plt.title("Related Words in Fake News vs. True News")
    red_patch = plt.plot([], [], marker="o", ls="",
                         mec=None, color="red", label="Fake News")
    blue_patch = plt.plot([], [], marker="o", ls="",
                          mec=None, color="blue", label="True News")
    yellow_patch = plt.plot([], [], marker="o", ls="",
                            mec=None, color="yellow", label="Longest Chain")
    plt.legend(handles=[red_patch[0], blue_patch[0], yellow_patch[0]])
    plt.show()
