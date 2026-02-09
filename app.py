import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

st.set_page_config(page_title="Clustering SMP Bekasi", layout="wide")

st.title("PENGELOMPOKAN KECAMATAN BERDASARKAN JUMLAH MURID SMP NEGERI DAN SWASTA MENGGUNAKAN ALGORITMA HIERARCHICAL CLUSTERING")
st.caption("Data Tahun 2021–2024 (BPS Kota Bekasi)")

uploaded = st.file_uploader(
    "Upload file CSV/Excel dengan kolom: Kecamatan, Rata_Negeri, Rata_Swasta",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Silakan upload file dulu ya 🙂")
    st.stop()

# --- baca file ---
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

st.subheader("Preview Data")
st.dataframe(df, use_container_width=True)

required_cols = {"Kecamatan", "Rata_Negeri", "Rata_Swasta"}
if not required_cols.issubset(set(df.columns)):
    st.error("Kolom belum sesuai. Pastikan ada: Kecamatan, Rata_Negeri, Rata_Swasta")
    st.stop()

# --- pastikan numerik ---
data = df.copy()
data["Rata_Negeri"] = pd.to_numeric(data["Rata_Negeri"], errors="coerce")
data["Rata_Swasta"] = pd.to_numeric(data["Rata_Swasta"], errors="coerce")

if data[["Rata_Negeri", "Rata_Swasta"]].isnull().any().any():
    st.error("Ada nilai yang tidak bisa diubah menjadi angka (NaN). Cek datamu ya.")
    st.stop()

# --- pilih jumlah cluster ---
st.subheader("Pengaturan Clustering")
k = st.slider("Pilih jumlah cluster", 2, 5, 3, 1)

# --- normalisasi + linkage ---
X = data[["Rata_Negeri", "Rata_Swasta"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method="ward")
labels = fcluster(Z, t=k, criterion="maxclust")
data["Cluster"] = labels

# --- silhouette score ---
try:
    sil = silhouette_score(X_scaled, labels)
    st.write(f"Silhouette Score (k={k}): **{sil:.4f}**")
except Exception:
    st.write("Silhouette Score tidak bisa dihitung untuk konfigurasi ini.")

# --- hasil clustering ---
st.subheader("Hasil Clustering per Kecamatan")
data_sorted = data.sort_values("Cluster")
st.dataframe(data_sorted, use_container_width=True)

# download hasil clustering
st.download_button(
    "Download hasil clustering (CSV)",
    data_sorted.to_csv(index=False).encode("utf-8"),
    file_name="hasil_clustering_kecamatan.csv",
    mime="text/csv"
)

# --- dendrogram ---
st.subheader("Dendrogram (Ward Linkage)")
fig1 = plt.figure(figsize=(10, 4))
dendrogram(Z, labels=data["Kecamatan"].astype(str).values, leaf_rotation=90)
plt.xlabel("Kecamatan")
plt.ylabel("Jarak")
plt.tight_layout()
st.pyplot(fig1)

# --- grafik batang per kecamatan (stacked) ---
st.subheader("Grafik Batang per Kecamatan (Negeri + Swasta)")
fig2 = plt.figure(figsize=(12, 5))
plt.bar(data_sorted["Kecamatan"], data_sorted["Rata_Negeri"], label="Negeri")
plt.bar(
    data_sorted["Kecamatan"],
    data_sorted["Rata_Swasta"],
    bottom=data_sorted["Rata_Negeri"],
    label="Swasta"
)
plt.title("Rata-rata Jumlah Murid SMP per Kecamatan (2021–2024)")
plt.xlabel("Kecamatan")
plt.ylabel("Rata-rata Jumlah Murid")
plt.xticks(rotation=90)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# --- ringkasan per cluster ---
st.subheader("Rata-rata per Cluster")
cluster_mean = data.groupby("Cluster")[["Rata_Negeri", "Rata_Swasta"]].mean().round(2)
st.dataframe(cluster_mean, use_container_width=True)
