# 🎗️ Dashboard: Detecção de Casos Ambíguos — Câncer de Mama

Análise de clustering não supervisionado aplicada ao Dataset Wisconsin
Breast Cancer, com foco na identificação de biópsias morfologicamente
ambíguas, as quais representam casos de maior risco de erro diagnóstico.

---

## 📌 Sobre o Projeto

O dashboard aplica dois algoritmos de clustering sobre 30 biomarcadores
morfológicos extraídos de biópsias por agulha fina (FNA):

- **DBSCAN** — identifica outliers: amostras que não se encaixam no
  padrão morfológico de nenhum grupo diagnóstico
- **Fuzzy C-Means** — atribui graus de pertencimento contínuos a cada
  amostra, revelando os casos de maior incerteza na fronteira entre
  malignos e benignos

O cruzamento entre os dois algoritmos aponta as biópsias que são
simultaneamente atípicas e ambíguas — os candidatos de maior atenção
clínica.

---

## 🛠️ Tecnologias

- Python 3.11
- Streamlit
- Plotly
- Pandas / NumPy
- scikit-learn (DBSCAN, PCA, StandardScaler)
- scikit-fuzzy (Fuzzy C-Means)
- SciPy
- openpyxl

---

## 📁 Estrutura do Projeto
├── dash_cancer_mama_dbscan_fuzzy.py   
├── dataset_cancer_mama_02.xlsx        
├── requirements.txt                   
└── README.md

---

## ▶️ Como Rodar Localmente

```bash
# Clone o repositório
git clone https://github.com/SEU_USUARIO/dashboard-cancer-mama-dbscan-fuzzy.git

# Acesse a pasta
cd dashboard-cancer-mama-dbscan-fuzzy

# Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Rode o dashboard
streamlit run dash_cancer_mama_dbscan_fuzzy.py
```

---

## 🌐 Deploy

Acesse o dashboard em produção:
[https://seu-usuario-dbscan-fuzzy.streamlit.app](https://seu-usuario-dbscan-fuzzy.streamlit.app)

---

## 👩‍💻 Autora

Maria Eduarda da Cruz de Camargo  
[LinkedIn](https://linkedin.com/in/seu-perfil) · [GitHub](https://github.com/SEU_USUARIO)
