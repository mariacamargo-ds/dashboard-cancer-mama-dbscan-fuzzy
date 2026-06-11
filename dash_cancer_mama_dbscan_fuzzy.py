import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sklearn as skl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import skfuzzy as fuzz

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Câncer de Mama · Análise Tumoral",
    page_icon="🎗️",
    layout="wide",
)

# ──────────────────────────────────────────────
# MODO ESCURO
# ──────────────────────────────────────────────
modo_escuro = st.sidebar.toggle("🌙 Modo escuro", value=False)

if modo_escuro:
    BG_APP      = "#0f0f1a"
    BG_SIDEBAR  = "#1a1025"
    BG_CARD     = "#1e1030"
    BG_PLOT     = "#16102a"
    BG_INFOBOX  = "#2a1040"
    COR_BORDA   = "#5a2555"
    COR_TITULO  = "#f5c2d9"
    COR_TEXTO   = "#d4a0be"
    COR_CAPTION = "#7a5070"
    COR_GRID    = "#2a1040"
    COR_MET_V   = "#f9d4e8"
    COR_MET_L   = "#c77daa"
else:
    BG_APP      = "#fdf6f9"
    BG_SIDEBAR  = "#f7e8f0"
    BG_CARD     = "#ffffff"
    BG_PLOT     = "#fff8fb"
    BG_INFOBOX  = "#fff0f6"
    COR_BORDA   = "#e8b4cc"
    COR_TITULO  = "#7b1c4b"
    COR_TEXTO   = "#5a1a35"
    COR_CAPTION = "#9b4468"
    COR_GRID    = "#f0d0de"
    COR_MET_V   = "#7b1c4b"
    COR_MET_L   = "#9b4468"

# ──────────────────────────────────────────────
# ESTILOS GLOBAIS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Lato', sans-serif; }}
  .stApp {{ background-color: {BG_APP}; }}

  section[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR};
    border-right: 2px solid {COR_BORDA};
  }}

  h1, h2, h3 {{
    color: {COR_TITULO} !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    text-align: center !important;
  }}

  p, span, label {{ color: {COR_TEXTO}; }}

  [data-testid="metric-container"] {{
    background: {BG_CARD};
    border: 1px solid {COR_BORDA};
    border-left: 5px solid #c0396e;
    border-radius: 10px;
    padding: 16px !important;
    text-align: center;
  }}
  [data-testid="metric-container"] label {{
    color: {COR_MET_L} !important;
    font-size: 0.8rem !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COR_MET_V} !important;
    font-size: 1.8rem !important;
  }}

  .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid {COR_BORDA}; }}
  .stTabs [data-baseweb="tab"] {{ color: {COR_CAPTION}; font-weight: 600; }}
  .stTabs [aria-selected="true"] {{
    color: #c0396e !important;
    border-bottom: 3px solid #c0396e !important;
  }}

  .info-box {{
    background: {BG_INFOBOX};
    border-left: 4px solid #c0396e;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: {COR_TEXTO};
    margin-top: 8px;
    text-align: center;
  }}

  .cluster-box {{
    background: {BG_INFOBOX};
    border-left: 4px solid #c0396e;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.9rem;
    color: {COR_TEXTO};
    margin: 12px 0;
    line-height: 1.75;
  }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dataset_cancer_mama_02.xlsx")
    df = df[df["raio médio"] < 100].copy()
    df["diagnóstico_label"] = df["diagnóstico"].map({0: "Maligno", 1: "Benigno"})
    return df

df = carregar_dados()

FEATURES = [
    "raio médio", "textura média", "perímetro médio", "área média",
    "suavidade média", "compacidade média", "concavidade média",
    "pontos côncavos médios", "simetria média", "dimensão fractal média",
    "erro do raio", "erro da textura", "erro do perímetro", "erro da área",
    "erro da suavidade", "erro da compacidade", "erro da concavidade",
    "erro dos pontos côncavos", "erro da simetria", "erro da dimensão fractal",
    "pior raio", "pior textura", "pior perímetro", "pior área",
    "pior suavidade", "pior compacidade", "pior concavidade",
    "piores pontos côncavos", "pior simetria", "pior dimensão fractal",
]

COR_MAL = "#c0396e"
COR_BEN = "#a1d541"
CORES   = {"Maligno": COR_MAL, "Benigno": COR_BEN}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_PLOT,
    font=dict(family="Lato", color=COR_TEXTO, size=13),
    xaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA),
    yaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA),
    legend=dict(bgcolor=BG_CARD, bordercolor=COR_BORDA, borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ──────────────────────────────────────────────
# SIDEBAR — FILTRO
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h2 style='text-align:center;color:{COR_TITULO};'>🎗️ Câncer de Mama</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:{COR_TEXTO};'>Análise de Características Tumorais</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;font-weight:700;color:{COR_TITULO};'>Tipos de Diagnóstico</p>",
        unsafe_allow_html=True,
    )
    diag_sel = st.multiselect(
        "Selecione:",
        options=["Maligno", "Benigno"],
        default=["Maligno", "Benigno"],
    )
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;font-size:0.78rem;color:{COR_CAPTION};'>"
        "Dataset Wisconsin Breast Cancer · 569 amostras · 30 features</p>",
        unsafe_allow_html=True,
    )

dff = df[df["diagnóstico_label"].isin(diag_sel)].copy() if diag_sel else df.copy()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def titulo(texto):
    st.markdown(f"<h3>{texto}</h3>", unsafe_allow_html=True)

def infobox(texto):
    st.markdown(
        f"<div class='info-box'>ℹ️ <b>Informações sobre o Gráfico</b><br>{texto}</div>",
        unsafe_allow_html=True,
    )

def clusterbox(texto):
    st.markdown(f"<div class='cluster-box'>{texto}</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────
st.markdown(
    "<h1>🎗️ Câncer de Mama: Análise de Características Tumorais</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:{COR_TEXTO};'>"
    "Exploração de biomarcadores extraídos por imagem digital de biópsia por agulha fina (FNA).</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────
# CARDS
# ──────────────────────────────────────────────
n_total = len(dff)
n_mal   = (dff["diagnóstico_label"] == "Maligno").sum()
n_ben   = (dff["diagnóstico_label"] == "Benigno").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total de Amostras", f"{n_total}")
c2.metric("Casos Malignos",    f"{n_mal}",
          f"{n_mal/n_total*100:.1f}% do total" if n_total else "")
c3.metric("Casos Benignos",    f"{n_ben}",
          f"{n_ben/n_total*100:.1f}% do total" if n_total else "")

st.divider()

# ──────────────────────────────────────────────
# ABAS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Visão Geral |",
    "Textura Média |",
    "Área × Concavidade |",
    "Compacidade Média |",
    "Simetria × Concavidade |",
    "DBSCAN + Fuzzy",
])

# ════════════════════════════════════════════
# TAB 1 · VISÃO GERAL
# ════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        titulo("Área Média por Diagnóstico")
        media_area = (
            dff.groupby("diagnóstico_label")["área média"]
            .mean().reset_index()
            .rename(columns={"área média": "Área Média (mm²)",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig1 = px.bar(
            media_area, x="Diagnóstico", y="Área Média (mm²)",
            color="Diagnóstico", color_discrete_map=CORES, text_auto=".0f",
        )
        fig1.update_traces(textposition="outside", textfont_size=13)
        fig1.update_layout(**LAYOUT_BASE, height=360, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        infobox("Tumores malignos apresentam área média significativamente maior, "
                "refletindo crescimento celular desordenado e expansão tumoral.")

    with col_b:
        titulo("Concavidade Média por Diagnóstico")
        media_conc = (
            dff.groupby("diagnóstico_label")["concavidade média"]
            .mean().reset_index()
            .rename(columns={"concavidade média": "Concavidade Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig2 = px.bar(
            media_conc, x="Diagnóstico", y="Concavidade Média",
            color="Diagnóstico", color_discrete_map=CORES, text_auto=".4f",
        )
        fig2.update_traces(textposition="outside", textfont_size=13)
        fig2.update_layout(**LAYOUT_BASE, height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        infobox("A concavidade descreve a severidade das reentrâncias no contorno celular. "
                "Valores mais altos indicam bordas irregulares, típicas de células malignas.")

# ════════════════════════════════════════════
# TAB 2 · TEXTURA MÉDIA
# ════════════════════════════════════════════
with tab2:
    titulo("Textura Média por Tipos de Diagnóstico")
    media_tex = (
        dff.groupby("diagnóstico_label")["textura média"]
        .mean().reset_index()
        .rename(columns={"textura média": "Textura Média",
                         "diagnóstico_label": "Diagnóstico"})
    )
    fig3 = px.bar(
        media_tex, x="Diagnóstico", y="Textura Média",
        color="Diagnóstico", color_discrete_map=CORES, text_auto=".2f",
    )
    fig3.update_traces(textposition="outside", textfont_size=14)
    fig3.update_layout(**LAYOUT_BASE, height=420, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    infobox("A textura mede o desvio-padrão dos valores de escala de cinza na imagem. "
            "Tecidos malignos tendem a apresentar textura mais heterogênea, "
            "decorrente da variabilidade nuclear elevada.")

# ════════════════════════════════════════════
# TAB 3 · ÁREA × CONCAVIDADE
# ════════════════════════════════════════════
with tab3:
    titulo("Área × Concavidade por Diagnóstico")
    fig4 = px.scatter(
        dff, x="área média", y="concavidade média",
        color="diagnóstico_label", color_discrete_map=CORES,
        labels={
            "área média": "Área Média (mm²)",
            "concavidade média": "Concavidade Média",
            "diagnóstico_label": "Diagnóstico",
        },
        opacity=0.75,
        hover_data={"área média": ":.1f", "concavidade média": ":.4f"},
    )
    fig4.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#fff")))
    fig4.update_layout(**LAYOUT_BASE, height=460)
    st.plotly_chart(fig4, use_container_width=True)
    infobox("A dispersão revela que casos malignos concentram-se em regiões de maior área "
            "<i>e</i> maior concavidade — as duas variáveis atuam juntas como marcadores "
            "de agressividade tumoral.")

# ════════════════════════════════════════════
# TAB 4 · COMPACIDADE MÉDIA
# ════════════════════════════════════════════
with tab4:
    titulo("Compacidade e Textura Média por Diagnóstico")
    col_c, col_d = st.columns(2)

    with col_c:
        media_comp = (
            dff.groupby("diagnóstico_label")["compacidade média"]
            .mean().reset_index()
            .rename(columns={"compacidade média": "Compacidade Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig5a = px.bar(
            media_comp, x="Diagnóstico", y="Compacidade Média",
            color="Diagnóstico", color_discrete_map=CORES,
            text_auto=".4f", title="Compacidade Média",
        )
        fig5a.update_traces(textposition="outside", textfont_size=13)
        fig5a.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                            title_x=0.5, title_font_color=COR_TITULO)
        st.plotly_chart(fig5a, use_container_width=True)

    with col_d:
        media_tex2 = (
            dff.groupby("diagnóstico_label")["textura média"]
            .mean().reset_index()
            .rename(columns={"textura média": "Textura Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig5b = px.bar(
            media_tex2, x="Diagnóstico", y="Textura Média",
            color="Diagnóstico", color_discrete_map=CORES,
            text_auto=".2f", title="Textura Média",
        )
        fig5b.update_traces(textposition="outside", textfont_size=13)
        fig5b.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                            title_x=0.5, title_font_color=COR_TITULO)
        st.plotly_chart(fig5b, use_container_width=True)

    infobox("A compacidade combina perímetro e área para medir a irregularidade do contorno. "
            "Quando associada à textura elevada, forma um par diagnóstico de alta relevância clínica.")

# ════════════════════════════════════════════
# TAB 5 · SIMETRIA × CONCAVIDADE
# ════════════════════════════════════════════
with tab5:
    titulo("Simetria Média × Concavidade Média por Diagnóstico")
    fig6 = px.scatter(
        dff, x="simetria média", y="concavidade média",
        color="diagnóstico_label", color_discrete_map=CORES,
        labels={
            "simetria média": "Simetria Média",
            "concavidade média": "Concavidade Média",
            "diagnóstico_label": "Diagnóstico",
        },
        opacity=0.75,
        hover_data={"simetria média": ":.4f", "concavidade média": ":.4f"},
    )
    fig6.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#fff")))
    fig6.update_layout(**LAYOUT_BASE, height=460)
    st.plotly_chart(fig6, use_container_width=True)
    infobox("Células malignas frequentemente exibem assimetria nuclear combinada com alta "
            "concavidade. A perda de simetria é um marcador morfológico clássico de "
            "malignidade tecidual.")

# ════════════════════════════════════════════
# TAB 6 · DBSCAN 3D + FUZZY C-MEANS
# ════════════════════════════════════════════
with tab6:
    st.markdown(
        "<h2>🔍 Detecção de Casos Ambíguos — DBSCAN 3D + Fuzzy C-Means</h2>",
        unsafe_allow_html=True,
    )

    clusterbox("""
    <b>O que esta análise faz?</b><br><br>
    Dois algoritmos de clustering não supervisionado trabalham em conjunto para identificar
    as amostras mais difíceis de classificar — aquelas que vivem na fronteira entre maligno
    e benigno, onde erros diagnósticos têm maior probabilidade de ocorrer.<br><br>
    O <b>DBSCAN</b> agrupa os dados por densidade no espaço 3D das três componentes principais
    (PCA) e rotula como <i>ruído</i> os pontos que não pertencem a nenhum cluster denso.
    Esses outliers são candidatos naturais a casos atípicos ou ambíguos.<br><br>
    O <b>Fuzzy C-Means</b> atribui a cada amostra um grau de pertencimento entre 0 e 1 para
    cada cluster. Amostras com pertencimento próximo de 0,5 para os dois grupos estão na zona
    de maior incerteza — são os casos onde a fronteira diagnóstica é mais tênue e o risco de
    falso positivo ou falso negativo é mais elevado.
    """)

    st.divider()

    # ── PARÂMETROS ───────────────────────────────
    st.markdown("<h3>⚙️ Parâmetros</h3>", unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        eps = st.slider(
            "DBSCAN · Eps (raio de vizinhança)",
            min_value=0.3, max_value=3.0, value=1.2, step=0.1,
            help="Raio máximo para considerar dois pontos como vizinhos. "
                 "Valores menores criam clusters mais densos e mais ruído.",
        )
    with col_p2:
        min_samples = st.slider(
            "DBSCAN · Min. amostras por cluster",
            min_value=2, max_value=20, value=5, step=1,
            help="Mínimo de pontos dentro do raio Eps para formar um cluster.",
        )
    with col_p3:
        limiar_fuzzy = st.slider(
            "Fuzzy · Limiar de ambiguidade",
            min_value=0.30, max_value=0.49, value=0.40, step=0.01,
            help="Amostras com pertencimento entre (limiar) e (1 - limiar) "
                 "para os dois clusters são consideradas ambíguas.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PRÉ-PROCESSAMENTO ────────────────────────
    X       = df[FEATURES].values
    y_label = df["diagnóstico_label"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # PCA → 3 componentes para o DBSCAN e visualização 3D
    pca  = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X_sc)
    var_exp = pca.explained_variance_ratio_ * 100

    # ── DBSCAN ───────────────────────────────────
    dbscan    = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = dbscan.fit_predict(X_3d)

    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_ruido       = int((labels_db == -1).sum())

    df_3d = pd.DataFrame({
        "PC1": X_3d[:, 0],
        "PC2": X_3d[:, 1],
        "PC3": X_3d[:, 2],
        "Diagnóstico Real":  y_label,
        "Cluster DBSCAN":    [f"Cluster {l}" if l != -1 else "⚠️ Ruído / Outlier"
                              for l in labels_db],
        "É Ruído":           labels_db == -1,
    })

    # ── FUZZY C-MEANS ────────────────────────────
    # skfuzzy espera dados com shape (features, amostras) — transpõe
    X_fuzzy = X_3d.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_fuzzy,
        c=2,          # 2 clusters: um por classe
        m=2.0,        # fuzziness — quanto maior, mais suaves as fronteiras
        error=0.005,
        maxiter=1000,
        init=None,
        seed=42,
    )

    # u[0] = pertencimento ao cluster 0, u[1] = cluster 1
    membro_0   = u[0]   # grau de pertencimento ao cluster 0
    membro_1   = u[1]   # grau de pertencimento ao cluster 1
    ambiguo    = (membro_0 > limiar_fuzzy) & (membro_0 < (1 - limiar_fuzzy))
    n_ambiguo  = int(ambiguo.sum())

    df_3d["Pertencimento C0"] = np.round(membro_0, 3)
    df_3d["Pertencimento C1"] = np.round(membro_1, 3)
    df_3d["Ambíguo (Fuzzy)"]  = ambiguo
    df_3d["Status Fuzzy"]     = np.where(ambiguo, "⚠️ Ambíguo", "✅ Definido")

    # ── CARDS DE RESUMO ───────────────────────────
    st.markdown("<h3>📊 Resumo dos Algoritmos</h3>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters DBSCAN",    f"{n_clusters_db}")
    m2.metric("Outliers DBSCAN",    f"{n_ruido}",
              help="Pontos que não pertencem a nenhum cluster denso.")
    m3.metric("Casos Ambíguos (Fuzzy)", f"{n_ambiguo}",
              help="Amostras com grau de pertencimento próximo de 0,5 para os dois clusters.")
    m4.metric("Variância explicada (PCA 3D)",
              f"{sum(var_exp):.1f}%",
              help="Proporção da variância total dos dados capturada pelas 3 componentes.")

    st.divider()

    # ── GRÁFICO 1: DBSCAN 3D ─────────────────────
    col_db, col_fz = st.columns(2)

    with col_db:
        titulo("DBSCAN · Clusters no Espaço 3D (PCA)")

        # Paleta de clusters — ruído sempre cinza
        cluster_ids  = sorted(set(labels_db))
        paleta_base  = ["#c0396e", "#a1d541", "#5b9bd5", "#f4a261",
                        "#9b5de5", "#00bbf9", "#f15bb5", "#fee440"]
        cor_cluster  = {}
        idx_cor      = 0
        for cid in cluster_ids:
            if cid == -1:
                cor_cluster["⚠️ Ruído / Outlier"] = "#888888"
            else:
                cor_cluster[f"Cluster {cid}"] = paleta_base[idx_cor % len(paleta_base)]
                idx_cor += 1

        fig_db = px.scatter_3d(
            df_3d,
            x="PC1", y="PC2", z="PC3",
            color="Cluster DBSCAN",
            color_discrete_map=cor_cluster,
            symbol="Diagnóstico Real",
            opacity=0.80,
            labels={
                "PC1": f"PC1 ({var_exp[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp[1]:.1f}%)",
                "PC3": f"PC3 ({var_exp[2]:.1f}%)",
            },
            hover_data={
                "Diagnóstico Real": True,
                "Cluster DBSCAN":   True,
                "PC1": ":.2f",
                "PC2": ":.2f",
                "PC3": ":.2f",
            },
        )
        fig_db.update_traces(marker=dict(size=4))
        fig_db.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor=BG_PLOT,
                xaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
                yaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
                zaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
            ),
            legend=dict(bgcolor=BG_CARD, bordercolor=COR_BORDA, borderwidth=1,
                        font_color=COR_TEXTO),
            font=dict(family="Lato", color=COR_TEXTO, size=12),
            margin=dict(l=0, r=0, t=30, b=0),
            height=480,
        )
        st.plotly_chart(fig_db, use_container_width=True)
        infobox(
            "Pontos cinzas são <b>outliers</b> — amostras que o DBSCAN não conseguiu "
            "associar a nenhum cluster denso. Explore a nuvem 3D girando o gráfico: "
            "os outliers tendem a se concentrar nas fronteiras entre os grupos diagnósticos."
        )

    # ── GRÁFICO 2: FUZZY 3D ──────────────────────
    with col_fz:
        titulo("Fuzzy C-Means · Grau de Pertencimento ao Cluster 0")

        fig_fz = px.scatter_3d(
            df_3d,
            x="PC1", y="PC2", z="PC3",
            color="Pertencimento C0",
            symbol="Status Fuzzy",
            color_continuous_scale=[
                [0.0,  "#5b9bd5"],   # azul → cluster benigno
                [0.5,  "#f0e0f0"],   # branco-rosado → zona ambígua
                [1.0,  "#c0396e"],   # rosa → cluster maligno
            ],
            opacity=0.85,
            labels={
                "PC1": f"PC1 ({var_exp[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp[1]:.1f}%)",
                "PC3": f"PC3 ({var_exp[2]:.1f}%)",
                "Pertencimento C0": "Pertencimento C0",
            },
            hover_data={
                "Diagnóstico Real":  True,
                "Status Fuzzy":      True,
                "Pertencimento C0":  ":.3f",
                "Pertencimento C1":  ":.3f",
                "PC1": ":.2f",
                "PC2": ":.2f",
                "PC3": ":.2f",
            },
        )
        fig_fz.update_traces(marker=dict(size=4))
        fig_fz.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor=BG_PLOT,
                xaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
                yaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
                zaxis=dict(backgroundcolor=BG_PLOT, gridcolor=COR_GRID,
                           tickfont_color=COR_TEXTO),
            ),
            coloraxis_colorbar=dict(
                title="Pertenc. C0",
                tickfont_color=COR_TEXTO,
                title_font_color=COR_TEXTO,
            ),
            legend=dict(bgcolor=BG_CARD, bordercolor=COR_BORDA, borderwidth=1,
                        font_color=COR_TEXTO),
            font=dict(family="Lato", color=COR_TEXTO, size=12),
            margin=dict(l=0, r=0, t=30, b=0),
            height=480,
        )
        st.plotly_chart(fig_fz, use_container_width=True)
        infobox(
            "A escala de cor vai de <b>azul</b> (pertence claramente ao cluster 0) a "
            "<b>rosa</b> (pertence claramente ao cluster 1). Os pontos em <b>tons claros "
            "intermediários</b> são os casos ambíguos — aqueles marcados como ⚠️ Ambíguo "
            "no símbolo — onde o Fuzzy não consegue decidir com confiança."
        )

    st.divider()

    # ── GRÁFICO 3: HISTOGRAMA DE PERTENCIMENTO ───
    titulo("Distribuição do Grau de Pertencimento ao Cluster 0 · por Diagnóstico Real")

    fig_hist = px.histogram(
        df_3d,
        x="Pertencimento C0",
        color="Diagnóstico Real",
        color_discrete_map={"Maligno": COR_MAL, "Benigno": COR_BEN},
        nbins=40,
        barmode="overlay",
        opacity=0.75,
        labels={
            "Pertencimento C0": "Grau de Pertencimento ao Cluster 0",
            "Diagnóstico Real": "Diagnóstico",
        },
    )
    # Linha vertical na zona de ambiguidade
    fig_hist.add_vline(
        x=limiar_fuzzy, line_dash="dash", line_color=COR_CAPTION,
        annotation_text=f"Limiar inferior ({limiar_fuzzy})",
        annotation_font_color=COR_CAPTION,
    )
    fig_hist.add_vline(
        x=1 - limiar_fuzzy, line_dash="dash", line_color=COR_CAPTION,
        annotation_text=f"Limiar superior ({1-limiar_fuzzy:.2f})",
        annotation_font_color=COR_CAPTION,
    )
    fig_hist.update_layout(**LAYOUT_BASE, height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

    infobox(
        "Idealmente, os malignos se concentram próximos de 0 ou 1 e os benignos no extremo "
        "oposto. As amostras que caem <b>entre as linhas tracejadas</b> — a zona central — "
        "são as de maior ambiguidade diagnóstica e maior risco de erro de classificação."
    )

    st.divider()

    # ── TABELA DE CASOS AMBÍGUOS ─────────────────
    titulo(f"Casos Ambíguos Identificados pelo Fuzzy C-Means  ·  {n_ambiguo} amostras")

    df_ambiguos = df.copy()
    df_ambiguos["Pertencimento C0"] = np.round(membro_0, 3)
    df_ambiguos["Pertencimento C1"] = np.round(membro_1, 3)
    df_ambiguos["Outlier DBSCAN"]   = np.where(labels_db == -1, "⚠️ Sim", "Não")
    df_ambiguos = df_ambiguos[ambiguo][[
        "diagnóstico_label", "área média", "concavidade média",
        "textura média", "compacidade média",
        "Pertencimento C0", "Pertencimento C1", "Outlier DBSCAN",
    ]].rename(columns={
        "diagnóstico_label": "Diagnóstico Real",
        "área média":        "Área Média",
        "concavidade média": "Concavidade Média",
        "textura média":     "Textura Média",
        "compacidade média": "Compacidade Média",
    }).reset_index(drop=True)

    st.dataframe(df_ambiguos, use_container_width=True, height=320)

    clusterbox(
        "<b>Como interpretar a tabela:</b> cada linha é uma amostra que o Fuzzy C-Means "
        "considerou ambígua — ou seja, cujo grau de pertencimento não se inclinou "
        "claramente para nenhum dos dois clusters. A coluna <b>Outlier DBSCAN</b> indica "
        "se o DBSCAN também classificou essa amostra como ponto de ruído, reforçando a "
        "suspeita de atipicidade. Amostras que aparecem como ambíguas no Fuzzy "
        "<i>e</i> como outliers no DBSCAN são os casos de maior atenção diagnóstica."
    )

# ──────────────────────────────────────────────
# RODAPÉ
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='text-align:center;font-size:0.85rem;color:{COR_CAPTION};'>"
    "🎗️ Dataset Wisconsin Breast Cancer | Análise exploratória | "
    "Streamlit + Plotly + scikit-learn + scikit-fuzzy</p>",
    unsafe_allow_html=True,
)
