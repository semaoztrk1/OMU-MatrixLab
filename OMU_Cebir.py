import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="OMÜ MatrixLab Web", page_icon="🧪", layout="wide")

# --- LOGO VE BAŞLIK ---
# omu_logo.png dosyasının bu script ile aynı klasörde olması gerekir
col_l, col_r = st.columns([1, 4])
with col_l:
    if os.path.exists("omu_logo.png"):
        logo = Image.open("omu_logo.png")
        st.image(logo, width=120)
with col_r:
    st.title("OMÜ Kimya Mühendisliği")
    st.subheader("Lineer Cebir Analiz Sistemi (MatrixLab Web)")

# --- SIDEBAR (KONTROL PANELİ) ---
st.sidebar.header("⚙️ Sistem Ayarları")
n = st.sidebar.number_input("Matris Boyutu (N)", min_value=2, max_value=10, value=3)
method = st.sidebar.selectbox("Çözüm Yöntemi", [
    "LU Doolittle", "Cholesky", "Gauss Yok Etme", "Gauss-Jordan", 
    "Cramer", "Jacobi", "Gauss-Seidel", "Gram-Schmidt (QR)"
])

st.sidebar.divider()
tol = st.sidebar.text_input("Tolerans", "0.0001")
max_it = st.sidebar.number_input("Maks. İterasyon", value=100)

# --- MATEMATİKSEL FONKSİYONLAR ---
def forward_sub(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def back_sub(U, y):
    x = np.zeros_like(y)
    for i in range(len(y)-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# --- ANA GİRİŞ ALANI ---
st.write("### 📝 Sistem Denklemleri ($Ax = B$)")
st.info("Kutucuklara tıklayarak değerleri girin. Boş bırakılan yerler 0.0 kabul edilir.")

# A ve B Matrisi için veri girişi (Streamlit data_editor çok pratiktir)
col_a, col_b = st.columns([3, 1])
with col_a:
    st.write("**Matris A (Katsayılar)**")
    matrix_a = st.data_editor(pd.DataFrame(np.zeros((n, n))), hide_index=True, key="a_input")

with col_b:
    st.write("**Vektör B**")
    vector_b = st.data_editor(pd.DataFrame(np.zeros((n, 1)), columns=["B"]), hide_index=True, key="b_input")

if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
    try:
        A = matrix_a.to_numpy()
        B = vector_b.to_numpy().flatten()
        log_messages = []

        # Çözüm Motoru
        if method == "LU Doolittle":
            L = np.eye(n); U = np.zeros((n, n))
            for i in range(n):
                for k in range(i, n): U[i, k] = A[i, k] - np.dot(L[i, :i], U[:i, k])
                for k in range(i + 1, n): L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]
            log_messages.append(f"L Matrisi:\n{L}")
            log_messages.append(f"U Matrisi:\n{U}")
            x = back_sub(U, forward_sub(L, B))

        elif method == "Cholesky":
            L = np.linalg.cholesky(A)
            x = back_sub(L.T, forward_sub(L, B))
            log_messages.append(f"Cholesky L:\n{L}")

        else:
            x = np.linalg.solve(A, B)
            log_messages.append("Sistem standart Numpy motoru ile çözüldü.")

        # --- SONUÇLARIN GÖSTERİLMESİ ---
        st.divider()
        res_col, log_col = st.columns([4, 6])
        
        with res_col:
            st.success("✅ Çözüm Bulundu")
            res_df = pd.DataFrame({"Xi": [f"x{i+1}" for i in range(n)], "Değer": x})
            st.table(res_df)
            
            # Excel İndirme Butonu
            excel_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Sonuçları İndir (CSV)", excel_data, "cozum.csv", "text/csv")

        with log_col:
            st.write("**📑 İşlem Basamakları**")
            for msg in log_messages:
                st.code(msg)

        # --- GRAFİK ---
        st.subheader("📊 Değer Dağılım Grafiği")
        fig, ax = plt.subplots()
        ax.bar(res_df["Xi"], res_df["Değer"], color='#2E86C1')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Hata: {e}")