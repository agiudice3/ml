import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.patches import Ellipse


def silhouette_plot(X, n_clusters, algorithm="kmeans", dataset_name="Dataset"):
    fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(7, 7)

    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-0.1, 1])

    # Initialize the clusterer
    if algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    elif algorithm == "gmm":
        clusterer = GaussianMixture(n_components=n_clusters, random_state=10)

    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        f"For {algorithm} with {n_clusters} clusters in {dataset_name}, the average silhouette_score is: {silhouette_avg}"
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples gap

    ax1.set_title(f"Silhouette plot for {dataset_name} using {algorithm}", fontsize=16)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=14)
    ax1.set_ylabel("Cluster label", fontsize=14)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.tight_layout()
    plt.show()


def project_centers_to_tsne_space(X_tsne, labels, n_clusters):
    centers = np.zeros((n_clusters, X_tsne.shape[1]))
    for i in range(n_clusters):
        centers[i, :] = X_tsne[labels == i].mean(axis=0)
    return centers


def plot_tsne_clusters(
    X_original,
    X_tsne,
    true_labels,
    title,
    filename,
    n_clusters=3,
    n_components=3,
    random_state=42,
):
    # Clustering on original data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(
        X_original
    )
    gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(
        X_original
    )

    kmeans_labels = kmeans.predict(X_original)
    gmm_labels = gmm.predict(X_original)

    # Project cluster centers into the t-SNE space
    kmeans_centers_tsne = project_centers_to_tsne_space(
        X_tsne, kmeans_labels, n_clusters
    )
    gmm_centers_tsne = project_centers_to_tsne_space(X_tsne, gmm_labels, n_clusters)

    # Plotting
    sns.set(
        context="paper",
        style="darkgrid",
        palette="muted",
        font="sans-serif",
        font_scale=1.2,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define color palette
    palette = sns.color_palette("husl", 3)

    # Ground truth clusters
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=true_labels,
        style=true_labels,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Ground Truth Clusters")
    axes[0].set_xlabel("Comp. 1")
    axes[0].set_ylabel("Comp. 2")
    axes[0].tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # KMeans clusters
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=kmeans_labels,
        style=kmeans_labels,
        palette=palette,
        markers=["o", "s"],
        ax=axes[1],
    )
    axes[1].scatter(
        kmeans_centers_tsne[:, 0],
        kmeans_centers_tsne[:, 1],
        s=100,
        c="black",
        marker="X",
        label="Centers",
    )

    axes[1].set_title("KMeans Clusters")
    axes[1].set_xlabel("Comp. 1")
    axes[1].set_ylabel("Comp. 2")
    axes[1].tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # GMM clusters
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=gmm_labels,
        style=gmm_labels,
        palette=palette,
        markers=["o", "s"],
        ax=axes[2],
    )
    axes[2].scatter(
        gmm_centers_tsne[:, 0],
        gmm_centers_tsne[:, 1],
        s=100,
        c="black",
        marker="X",
        label="Centers",
    )

    axes[2].set_title("GMM Clusters")
    axes[2].set_xlabel("Comp. 1")
    axes[2].set_ylabel("Comp. 2")
    axes[2].tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # Show legends
    for ax in axes:
        ax.legend(loc="upper right")

    # Adjust layout and save
    plt.suptitle(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        e = Ellipse(
            xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs
        )
        ax.add_patch(e)
    return ax


def plot_pca_results(X_pca, true_labels, gmm_labels, gmm, title, filename):
    palette = sns.color_palette("husl", 3)
    sns.set(
        context="paper",
        style="darkgrid",
        palette="muted",
        font="sans-serif",
        font_scale=1.75,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot ground truth labels
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=true_labels, palette=palette, ax=ax1
    )
    ax1.set_title(f"{title} Ground Truth", fontsize=16)
    ax1.set_xlabel("Component 1", fontsize=14)
    ax1.set_ylabel("Component 2", fontsize=14)
    ax1.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # Plot GMM cluster labels
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=gmm_labels, palette=palette, ax=ax2
    )
    ax2.set_title(f"{title} GMM Clusters", fontsize=16)
    ax2.set_xlabel("Component 1", fontsize=14)
    ax2.set_ylabel("Component 2", fontsize=14)
    ax2.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # Add GMM confidence ellipses
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax2)

    for ax in (ax1, ax2):
        ax.legend(loc="upper right")

    plt.suptitle(title, fontsize=16)
    plt.savefig(filename)
    plt.show()
