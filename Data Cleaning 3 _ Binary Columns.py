import os
import pandas as pd
import numpy as np
import umap
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.datasets import make_blobs
from sklearn.random_projection import GaussianRandomProjection
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Step01: Connecting 7 OHLCV CSV files into the project
file_path = 'D:/OneDrive - The Pennsylvania State University/01 DAAN 545/00 Project/00 Data Source/Week3_Raw_Data/Sorting_Printing_Gap/Data_with_3_Printing_Gap.csv'

# Step02: Read and show the CSV file with encoding='iso-8859-1'
df_bg = pd.read_csv(file_path, encoding='iso-8859-1')

# Filter the data based on the condition
df_bg = df_bg[df_bg['Number_of_use_Printing_Cylinder'] == 3]

df_bg['Rejection_Result'] = np.where(
    (df_bg['BCT_Avg'] < df_bg['BCT_Spec']) | (abs((df_bg['BCT_Avg'] - df_bg['BCT_Spec']) / df_bg['BCT_Spec']) < 0.03),
    0, 1
)

filtered_df = df_bg[df_bg['Rejection_Result'] == 1].head(15000)  # Ensure to select only the first 1000 rows

selected_columns = ["Raw_Mat_Strength_Type", "Raw_Mat_Strength_Additive", "BCT_Avg", "ECT_Avg", "Feed_Roll_Gap", "New_Printing_Gap_1", "New_Printing_Gap_2", "New_Printing_Gap_3", "Speed_Act"]
df_filtered = filtered_df.loc[:, selected_columns]

print("Test Reading the Dataset")
print(df_filtered)
print(df_filtered.dtypes)
print('\n' + '-' * 50 + '\n')  # Add a separator between columns

print("Test Read Dataset")
print('\n' + '-' * 50 + '\n')  # Add a separator between columns
print(df_filtered)

# Normalize
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(df_filtered.iloc[:, 2:]), columns=df_filtered.columns[2:])
print(data)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
pca_data = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
print("Explained Variance Ratio for PCA:")
print(pca.explained_variance_ratio_)

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True)
kpca_data = kpca.fit_transform(data)
kpca_data = pd.DataFrame(kpca_data, columns=['PC1', 'PC2'])
print("Explained Variance for Kernel PCA:")
explained_variance_kpca = np.var(kpca_data, axis=0)
explained_variance_ratio_kpca = explained_variance_kpca / np.sum(explained_variance_kpca)
print(explained_variance_ratio_kpca)

# Truncated SVD
svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(data)
svd_data = pd.DataFrame(svd_data, columns=['PC1', 'PC2'])
print("Explained Variance Ratio for Truncated SVD:")
print(svd.explained_variance_ratio_)

# NMF
nmf = NMF(n_components=2)
nmf_data = nmf.fit_transform(data + abs(data.min().min()))
nmf_data = pd.DataFrame(nmf_data, columns=['PC1', 'PC2'])
print("Explained Variance for NMF:")
explained_variance_nmf = np.var(nmf_data, axis=0)
explained_variance_ratio_nmf = explained_variance_nmf / np.sum(explained_variance_nmf)
print(explained_variance_ratio_nmf)

# ICA
ica = FastICA(n_components=2)
ica_data = ica.fit_transform(data)
ica_data = pd.DataFrame(ica_data, columns=['PC1', 'PC2'])
explained_variance_ica = np.var(ica_data, axis=0)
explained_variance_ratio_ica = explained_variance_ica / np.sum(explained_variance_ica)
print("Explained Variance for ICA:")
print(explained_variance_ratio_ica)

# UMAP
umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(data)
umap_data = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'])
print("Explained Variance Ratio for UMAP:")
print("UMAP does not have an explained variance ratio.")

# t-SNE
tsne = TSNE(n_components=2, random_state=1)
tsne_data = tsne.fit_transform(data)
tsne_data = pd.DataFrame(tsne_data, columns=['TSNE1', 'TSNE2'])
print("Explained Variance Ratio for t-SNE:")
print("t-SNE does not have an explained variance ratio.")

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=1)

# PCA + K-Means
kmeans.fit(pca_data)
pca_data['KMeansCluster'] = kmeans.labels_

# Kernel PCA + K-Means
kmeans.fit(kpca_data)
kpca_data['KMeansCluster'] = kmeans.labels_

# Truncated SVD + K-Means
kmeans.fit(svd_data)
svd_data['KMeansCluster'] = kmeans.labels_

# NMF + K-Means
kmeans.fit(nmf_data)
nmf_data['KMeansCluster'] = kmeans.labels_

# ICA + K-Means
kmeans.fit(ica_data)
ica_data['KMeansCluster'] = kmeans.labels_

# UMAP + K-Means
kmeans.fit(umap_data)
umap_data['KMeansCluster'] = kmeans.labels_

# t-SNE + K-Means
kmeans.fit(tsne_data)
tsne_data['KMeansCluster'] = kmeans.labels_

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)

# PCA + DBSCAN
dbscan.fit(pca_data[['PC1', 'PC2']])
pca_data['DBSCANCluster'] = dbscan.labels_

# Kernel PCA + DBSCAN
dbscan.fit(kpca_data[['PC1', 'PC2']])
kpca_data['DBSCANCluster'] = dbscan.labels_

# Truncated SVD + DBSCAN
dbscan.fit(svd_data[['PC1', 'PC2']])
svd_data['DBSCANCluster'] = dbscan.labels_

# NMF + DBSCAN
dbscan.fit(nmf_data[['PC1', 'PC2']])
nmf_data['DBSCANCluster'] = dbscan.labels_

# ICA + DBSCAN
dbscan.fit(ica_data[['PC1', 'PC2']])
ica_data['DBSCANCluster'] = dbscan.labels_

# UMAP + DBSCAN
dbscan.fit(umap_data[['UMAP1', 'UMAP2']])
umap_data['DBSCANCluster'] = dbscan.labels_

# t-SNE + DBSCAN
dbscan.fit(tsne_data[['TSNE1', 'TSNE2']])
tsne_data['DBSCANCluster'] = dbscan.labels_

# Plot scatter graphs
fig, axes = plt.subplots(4, 4, figsize=(15, 15))

# PCA
axes[0, 0].scatter(pca_data['PC1'], pca_data['PC2'], c=pca_data['KMeansCluster'])
axes[0, 0].set_title('PCA + K-Means')
axes[0, 1].scatter(pca_data['PC1'], pca_data['PC2'], c=pca_data['DBSCANCluster'])
axes[0, 1].set_title('PCA + DBSCAN')

# Kernel PCA
axes[0, 2].scatter(kpca_data['PC1'], kpca_data['PC2'], c=kpca_data['KMeansCluster'])
axes[0, 2].set_title('Kernel PCA + K-Means')
axes[0, 3].scatter(kpca_data['PC1'], kpca_data['PC2'], c=kpca_data['DBSCANCluster'])
axes[0, 3].set_title('Kernel PCA + DBSCAN')

# Truncated SVD
axes[1, 0].scatter(svd_data['PC1'], svd_data['PC2'], c=svd_data['KMeansCluster'])
axes[1, 0].set_title('T-SVD + K-Means')
axes[1, 1].scatter(svd_data['PC1'], svd_data['PC2'], c=svd_data['DBSCANCluster'])
axes[1, 1].set_title('T-SVD + DBSCAN')

# NMF
axes[1, 2].scatter(nmf_data['PC1'], nmf_data['PC2'], c=nmf_data['KMeansCluster'])
axes[1, 2].set_title('NMF + K-Means')
axes[1, 3].scatter(nmf_data['PC1'], nmf_data['PC2'], c=nmf_data['DBSCANCluster'])
axes[1, 3].set_title('NMF + DBSCAN')

# ICA
axes[2, 0].scatter(ica_data['PC1'], ica_data['PC2'], c=ica_data['KMeansCluster'])
axes[2, 0].set_title('ICA + K-Means')
axes[2, 1].scatter(ica_data['PC1'], ica_data['PC2'], c=ica_data['DBSCANCluster'])
axes[2, 1].set_title('ICA + DBSCAN')

# UMAP
axes[2, 2].scatter(umap_data['UMAP1'], umap_data['UMAP2'], c=umap_data['KMeansCluster'])
axes[2, 2].set_title('UMAP + K-Means')
axes[2, 3].scatter(umap_data['UMAP1'], umap_data['UMAP2'], c=umap_data['DBSCANCluster'])
axes[2, 3].set_title('UMAP + DBSCAN')

# t-SNE
axes[3, 0].scatter(tsne_data['TSNE1'], tsne_data['TSNE2'], c=tsne_data['KMeansCluster'])
axes[3, 0].set_title('t-SNE + K-Means')
axes[3, 1].scatter(tsne_data['TSNE1'], tsne_data['TSNE2'], c=tsne_data['DBSCANCluster'])
axes[3, 1].set_title('t-SNE + DBSCAN')

plt.tight_layout()
plt.show()

# UMAP
umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(data)

# t-SNE
tsne = TSNE(n_components=2, random_state=1)
tsne_data = tsne.fit_transform(data)

# Calculate density estimates for UMAP and t-SNE embeddings
umap_density = gaussian_kde(umap_data.T)
tsne_density = gaussian_kde(tsne_data.T)

# Generate grid for density estimation
x_umap, y_umap = np.mgrid[min(umap_data[:, 0]):max(umap_data[:, 0]):100j, min(umap_data[:, 1]):max(umap_data[:, 1]):100j]
x_tsne, y_tsne = np.mgrid[min(tsne_data[:, 0]):max(tsne_data[:, 0]):100j, min(tsne_data[:, 1]):max(tsne_data[:, 1]):100j]
positions_umap = np.vstack([x_umap.ravel(), y_umap.ravel()])
positions_tsne = np.vstack([x_tsne.ravel(), y_tsne.ravel()])
density_umap = np.reshape(umap_density(positions_umap).T, x_umap.shape)
density_tsne = np.reshape(tsne_density(positions_tsne).T, x_tsne.shape)

# Plot UMAP with density estimation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(density_umap), extent=[min(umap_data[:, 0]), max(umap_data[:, 0]), min(umap_data[:, 1]), max(umap_data[:, 1])], cmap=plt.cm.Blues, aspect='auto')
plt.scatter(umap_data[:, 0], umap_data[:, 1], c='red', s=5)
plt.title("UMAP Embedding with Density Estimation")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# Plot t-SNE with density estimation
plt.subplot(1, 2, 2)
plt.imshow(np.rot90(density_tsne), extent=[min(tsne_data[:, 0]), max(tsne_data[:, 0]), min(tsne_data[:, 1]), max(tsne_data[:, 1])], cmap=plt.cm.Blues, aspect='auto')
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='red', s=5)
plt.title("t-SNE Embedding with Density Estimation")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.show()