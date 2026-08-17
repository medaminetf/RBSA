"""RBSA-ML — Interface Streamlit (Africapital Management).

Analyse de style des OPCVM Actions marocains par apprentissage automatique
(adaptation ML de la RBSA de Sharpe), branchée sur les vraies données MASI/ASFIM.

Lancer :
    streamlit run app/streamlit_app.py
ou, sous Windows, double-cliquer sur "Lancer_application.bat" à la racine du projet.

Prérequis : avoir exécuté au moins une fois `python scripts/run_pipeline.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbsa.config import factor_names, load_config, macro_aggregation_matrix, set_seed
from rbsa.application.opcvm import analyze_fund
from rbsa.models.qp_baseline import rolling_qp
from rbsa.training.dataset import load_split
from rbsa.training.train import infer_sequence

# ------------------------------------------------------------------ thème
NAVY = "#1F3A5F"
NAVY_DARK = "#152A45"
GOLD = "#C79612"
BG = "#F7F6F3"
CARD = "#FFFFFF"
PALETTE = ["#2F5798", "#C79612", "#56A0DA", "#A64B32", "#3B9A5C",
           "#8A5BC7", "#D07030", "#1899AC", "#C24D74", "#90942B"]

st.set_page_config(
    page_title="RBSA-ML — Africapital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BG}; }}
      .stApp h1, .stApp h2, .stApp h3 {{ color: {NAVY}; }}
      div[data-testid="stMetric"] {{
          background: {CARD}; border-radius: 10px; padding: 14px 16px;
          border: 1px solid #e7e4dd;
      }}
      div[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
      div[data-testid="stMetricLabel"] {{ color: #6b6b6b; }}
      .stApp [data-testid="stHeader"] {{ background: transparent; }}
      div[data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
      .stTabs [aria-selected="true"] {{ color: {NAVY}; }}
      .rbsa-banner {{
          background: linear-gradient(90deg, {NAVY} 0%, {NAVY_DARK} 100%);
          color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px;
      }}
      .rbsa-banner h1 {{ color: white; margin: 0 0 4px 0; font-size: 1.5rem; }}
      .rbsa-banner p {{ color: #d8dee8; margin: 0; font-size: 0.95rem; }}
      .rbsa-pill {{
          display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 0.78rem; font-weight: 600; margin-right: 6px;
      }}
      .rbsa-pill-ok {{ background: #e3f3e6; color: #1e7a34; }}
      .rbsa-pill-warn {{ background: #fdeee0; color: #a05a12; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="rbsa-banner">
      <h1>📊 RBSA-ML — Analyse de style des OPCVM Actions</h1>
      <p>Où un fonds actions marocain investit-il réellement, secteur par secteur, semaine après semaine —
      calculé instantanément par un modèle entraîné sur les données MASI/ASFIM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================ chargement
@st.cache_resource(show_spinner="Chargement du modèle et des données…")
def load_everything():
    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg["seed"])
    factors = factor_names(cfg)
    dataset_dir = ROOT / cfg["paths"]["dataset"]
    models_dir = ROOT / cfg["paths"]["models"]
    if not dataset_dir.exists() or not (models_dir / "gru_best.pt").exists():
        return None

    test_pf, X_test, dates_test = load_split(dataset_dir, "test", factors)

    import torch
    from rbsa.models.networks import build_model

    ckpt = torch.load(models_dir / "gru_best.pt", map_location="cpu", weights_only=False)
    gru = build_model("gru", cfg, len(factors) + 1, len(factors))
    gru.load_state_dict(ckpt["state_dict"])
    gru.eval()

    X_df_full = pd.read_parquet(dataset_dir / "style_matrix.parquet")

    asfim_vl, asfim_meta = None, None
    rp = dataset_dir / "asfim_ACTIONS_returns.parquet"
    if rp.exists():
        asfim_vl = pd.read_parquet(rp)
        mp = dataset_dir / "asfim_ACTIONS_meta.parquet"
        asfim_meta = pd.read_parquet(mp) if mp.exists() else None

    return (cfg, factors, test_pf, X_test, dates_test, gru, ckpt["mean"], ckpt["std"],
            X_df_full, asfim_vl, asfim_meta)


loaded = load_everything()

if loaded is None:
    st.error("### Aucun modèle entraîné trouvé")
    st.markdown(
        """
        L'application a besoin qu'un modèle ait été entraîné au moins une fois avant de pouvoir l'utiliser.

        **Comment corriger ça :**

        1. Ouvre un terminal dans le dossier `rbsa_ml`.
        2. Installe les dépendances une seule fois : `pip install -r requirements.txt`
        3. Lance l'entraînement : `python scripts/run_pipeline.py`
           (compte environ 15 minutes ; `--quick` pour une version rapide de démonstration)
        4. Relance cette application.
        """
    )
    st.stop()

(cfg, factors, test_pf, X_test, dates_test, gru, f_mean, f_std,
 X_df_full, asfim_vl, asfim_meta) = loaded
A, macro_names = macro_aggregation_matrix(cfg)


@st.cache_data(show_spinner="Analyse de l'ensemble des fonds Actions…")
def compute_overview(_gru, _mean, _std, _X_df, _asfim_vl, _cfg, _factors):
    rows = []
    for fund in _asfim_vl.columns:
        y_ser = _asfim_vl[fund].dropna()
        dates = y_ser.index.intersection(_X_df.index)
        if len(dates) < _cfg["model"]["window"] + 5:
            continue
        y_ser = y_ser.loc[dates]
        X = _X_df.loc[dates].to_numpy()
        theta = infer_sequence(_gru, y_ser.to_numpy(), X, _mean, _std)
        rep = analyze_fund(y_ser, _X_df.loc[dates], theta, _cfg)
        rows.append({
            "Fonds": fund,
            "Semaines": len(dates),
            "R² réplication": rep["r2_replication"],
            "Alpha net (ann.)": rep["alpha_net_annuel"],
            "Poids actions moyen": rep["poids_actions_moyen"],
            "Tracking error (ann.)": rep["tracking_error_vol"],
            "Alertes": len(rep["alerts"]),
        })
    return pd.DataFrame(rows).sort_values("R² réplication", ascending=False).reset_index(drop=True)


overview_df = compute_overview(gru, f_mean, f_std, X_df_full, asfim_vl, cfg, factors) if asfim_vl is not None else None


# ============================================================ sidebar : sélection du fonds
with st.sidebar:
    st.markdown("### 🗂️ Fonds à analyser")

    source_options = []
    if asfim_vl is not None:
        source_options.append("Fonds ASFIM réel")
    source_options += ["Portefeuille test (démo)", "Charger un fichier (CSV/Excel)"]
    source = st.radio("Source des données", source_options, index=0, label_visibility="collapsed")

    uploaded, p, real_fund = None, None, None

    if source == "Fonds ASFIM réel" and asfim_vl is not None:
        n_obs = asfim_vl.count().sort_values(ascending=False)
        options = list(n_obs.index)
        default_idx = next((i for i, f in enumerate(options) if "AFRICAPITAL" in f.upper()), 0)
        real_fund = st.selectbox(
            f"Choisir un fonds ({len(options)} disponibles)", options, index=default_idx,
            help="Tape pour rechercher un fonds par nom.",
        )
    elif source == "Portefeuille test (démo)":
        labels = [f"{pp['pid']} · {pp['archetype']}" for pp in test_pf]
        choice = st.selectbox("Portefeuille du jeu de test", labels)
        p = test_pf[labels.index(choice)]
    else:
        uploaded = st.file_uploader(
            "Fichier avec Date + VL hebdomadaire", type=["csv", "xlsx"],
            help="Deux colonnes : une date, une valeur liquidative (VL).",
        )

    st.divider()
    with st.expander("⚙️ Options avancées"):
        granularite = st.radio("Niveau de détail sectoriel", ["Macro-secteurs (recommandé)", "24 secteurs MASI"], index=0)
        show_qp = st.checkbox("Comparer avec la méthode classique (QP)", value=False)

    st.divider()
    st.caption(
        "Modèle : GRU entraîné sur données réelles MASI/ASFIM 2013-2026. "
        "Les poids sont estimés, jamais garantis exacts — voir l'onglet Méthode."
    )


# ============================================================ résolution des données du fonds sélectionné
if real_fund is not None:
    y_ser = asfim_vl[real_fund].dropna()
    dates = y_ser.index.intersection(X_df_full.index)
    y_ser = y_ser.loc[dates]
    y, dates = y_ser.to_numpy(), y_ser.index
    X = X_df_full.loc[dates].to_numpy()
    fund_name = real_fund
elif uploaded is not None:
    from rbsa.data.loaders import _find_date_column

    try:
        raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith("x") else pd.read_csv(uploaded, sep=None, engine="python")
        dcol = _find_date_column(raw)
        vals = [c for c in raw.columns if c != dcol][0]
        vl = pd.Series(
            pd.to_numeric(raw[vals], errors="coerce").values,
            index=pd.to_datetime(raw[dcol], errors="coerce"),
        ).dropna().sort_index()
        y_ser = vl.pct_change().reindex(X_df_full.index).dropna()
        if len(y_ser) < cfg["model"]["window"] + 5:
            st.warning("Fichier chargé mais trop court pour une analyse fiable (minimum quelques mois de données hebdomadaires).")
            st.stop()
        y = y_ser.to_numpy()
        dates = y_ser.index
        X = X_df_full.loc[dates].to_numpy()
        fund_name = uploaded.name
    except Exception as exc:
        st.error(f"Impossible de lire ce fichier : {exc}")
        st.stop()
elif p is not None:
    y, dates, X = p["y"], dates_test, X_test
    fund_name = p["pid"]
else:
    st.info("👈 Choisis un fonds dans le panneau de gauche pour commencer.")
    st.stop()


# ============================================================ inférence (instantanée)
theta = infer_sequence(gru, y, X, f_mean, f_std)
rep = analyze_fund(pd.Series(y, index=dates), pd.DataFrame(X, index=dates, columns=factors), theta, cfg)

if granularite.startswith("Macro"):
    theta_plot = np.einsum("mk,tk->tm", A, theta)
    names_plot = macro_names
else:
    theta_plot, names_plot = theta, factors
df_w = pd.DataFrame(theta_plot, index=dates, columns=names_plot)


# ============================================================ onglets
tab_overview, tab_detail, tab_compare, tab_about = st.tabs(
    ["🏠 Vue d'ensemble", "🔍 Fiche du fonds", "⚖️ Comparer des fonds", "ℹ️ Comment ça marche"]
)

# ---------------------------------------------------------------- Vue d'ensemble
with tab_overview:
    if overview_df is None:
        st.info("Aucune donnée ASFIM réelle chargée — branche `data/asfim_hebdomadaire` pour activer cette vue.")
    else:
        st.markdown("#### Panorama des fonds Actions ASFIM")
        st.caption("Un coup d'œil sur tous les fonds analysés. Clique sur un en-tête de colonne pour trier.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fonds analysés", len(overview_df))
        c2.metric("R² médian", f"{overview_df['R² réplication'].median():.2f}")
        c3.metric("TE médian (ann.)", f"{overview_df['Tracking error (ann.)'].median():.1%}")
        c4.metric("Fonds avec alerte", int((overview_df["Alertes"] > 0).sum()))

        st.markdown("")
        show = overview_df.copy()
        show["R² réplication"] = show["R² réplication"].round(2)
        show["Alpha net (ann.)"] = show["Alpha net (ann.)"].apply(lambda v: f"{v:+.2%}")
        show["Poids actions moyen"] = show["Poids actions moyen"].apply(lambda v: f"{v:.0%}")
        show["Tracking error (ann.)"] = show["Tracking error (ann.)"].apply(lambda v: f"{v:.1%}")

        st.dataframe(
            show,
            width="stretch",
            height=520,
            hide_index=True,
            column_config={
                "R² réplication": st.column_config.NumberColumn(format="%.2f"),
                "Alertes": st.column_config.NumberColumn(format="%d ⚠️"),
            },
        )
        st.caption(
            "💡 Pour voir le détail d'un fonds (poids sectoriels dans le temps, alertes de conformité), "
            "sélectionne-le dans le panneau de gauche puis va dans l'onglet **Fiche du fonds**."
        )

# ---------------------------------------------------------------- Fiche du fonds
with tab_detail:
    st.markdown(f"#### {fund_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² de réplication", f"{rep['r2_replication']:.2f}",
              help="Part de la variation du fonds expliquée par les secteurs estimés. Plus proche de 1 = meilleure réplication.")
    c2.metric("Alpha net (annuel)", f"{rep['alpha_net_annuel']:+.2%}",
              help="Rendement du fonds au-delà de ce qu'expliquent ses expositions sectorielles.")
    c3.metric("Poids actions moyen", f"{rep['poids_actions_moyen']:.0%}",
              help="Doit rester ≥ 60% en permanence pour un fonds Actions (réglementation AMMC).")
    c4.metric("Tracking error (ann.)", f"{rep['tracking_error_vol']:.2%}",
              help="Écart type annualisé entre le fonds et sa réplication sectorielle.")

    st.markdown("")
    if rep["alerts"]:
        for a in rep["alerts"]:
            st.warning(f"**{a['type'].replace('_', ' ').title()}** — {a['message']}")
    else:
        st.success("✅ Aucune alerte de conformité (exposition ≥ 60 %, R² ≥ 0,70, pas de dérive de style anormale).")

    st.markdown("##### Répartition sectorielle dans le temps")
    order = df_w.mean().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, name in enumerate(order):
        fig.add_trace(go.Scatter(
            x=df_w.index, y=df_w[name], stackgroup="w", mode="lines",
            line=dict(width=0.6, color=PALETTE[i % len(PALETTE)]),
            fillcolor=PALETTE[i % len(PALETTE)],
            name=name,
            hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1], title=None, gridcolor="#eceae6"),
        xaxis=dict(title=None, gridcolor="#eceae6"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified", height=460, margin=dict(t=10),
        font=dict(color="#333"),
    )
    st.plotly_chart(fig, width="stretch")

    if show_qp:
        with st.spinner("Résolution de la méthode classique (QP) en fenêtre glissante…"):
            theta_qp = rolling_qp(y, X, window=cfg["qp_baseline"]["window"], lam=1e-3)
        theta_qp_plot = np.einsum("mk,tk->tm", A, theta_qp) if granularite.startswith("Macro") else theta_qp
        df_qp = pd.DataFrame(theta_qp_plot, index=dates, columns=names_plot)
        top = order[0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_w.index, y=df_w[top], name=f"{top} — modèle ML",
                                  line=dict(color=PALETTE[0], width=2)))
        fig2.add_trace(go.Scatter(x=df_qp.index, y=df_qp[top], name=f"{top} — méthode classique",
                                  line=dict(color=PALETTE[1], width=2, dash="dot")))
        fig2.update_layout(title=f"ML vs méthode classique — poids « {top} »",
                           yaxis=dict(tickformat=".0%", gridcolor="#eceae6"),
                           plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                           hovermode="x unified", height=340, margin=dict(t=40))
        st.plotly_chart(fig2, width="stretch")

    with st.expander("📋 Voir les poids estimés en tableau (et les exporter)"):
        st.dataframe(df_w.style.format("{:.1%}"), width="stretch")
        st.download_button(
            "⬇️ Télécharger en CSV",
            df_w.to_csv().encode("utf-8"),
            file_name=f"poids_sectoriels_{fund_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.caption("Poids estimés par le modèle ML (softmax : poids ≥ 0, somme = 1 par construction). "
               "Facteur CASH ≈ taux directeur BAM / 52.")

# ---------------------------------------------------------------- Comparer
with tab_compare:
    st.markdown("#### Comparer plusieurs fonds")
    if asfim_vl is None:
        st.info("Aucune donnée ASFIM réelle chargée — cette vue nécessite des fonds réels.")
    else:
        default_pick = [f for f in [fund_name] if f in asfim_vl.columns][:1]
        picks = st.multiselect(
            "Choisir jusqu'à 4 fonds à comparer", list(asfim_vl.columns),
            default=default_pick, max_selections=4,
        )
        if len(picks) < 2:
            st.info("Sélectionne au moins deux fonds pour lancer la comparaison.")
        else:
            comp_rows, comp_series = [], {}
            for f in picks:
                y_ser = asfim_vl[f].dropna()
                d = y_ser.index.intersection(X_df_full.index)
                y_ser = y_ser.loc[d]
                Xf = X_df_full.loc[d].to_numpy()
                th = infer_sequence(gru, y_ser.to_numpy(), Xf, f_mean, f_std)
                r = analyze_fund(y_ser, X_df_full.loc[d], th, cfg)
                comp_rows.append({
                    "Fonds": f, "R² réplication": round(r["r2_replication"], 2),
                    "Alpha net (ann.)": f"{r['alpha_net_annuel']:+.2%}",
                    "Tracking error (ann.)": f"{r['tracking_error_vol']:.1%}",
                    "Alertes": len(r["alerts"]),
                })
                th_macro = np.einsum("mk,tk->tm", A, th)
                comp_series[f] = pd.DataFrame(th_macro, index=d, columns=macro_names)

            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

            common_sector = st.selectbox("Comparer l'exposition à un macro-secteur", macro_names)
            fig3 = go.Figure()
            for i, f in enumerate(picks):
                s = comp_series[f][common_sector]
                fig3.add_trace(go.Scatter(x=s.index, y=s, name=f, mode="lines",
                                          line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
            fig3.update_layout(
                title=f"Exposition « {common_sector} » dans le temps",
                yaxis=dict(tickformat=".0%", gridcolor="#eceae6"),
                plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                hovermode="x unified", height=420, margin=dict(t=40),
            )
            st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------- À propos
with tab_about:
    st.markdown("""
#### Ce que fait cet outil

Un fonds actions marocain ne publie pas systématiquement le détail de ses positions sectorielles.
Cet outil **déduit** cette répartition à partir des seules variations hebdomadaires de la valeur
liquidative (VL) du fonds, en la comparant à l'évolution des 24 secteurs de la Bourse de Casablanca.

C'est l'adaptation par apprentissage automatique de l'analyse de style de William Sharpe (RBSA) :
un modèle a été entraîné sur des milliers de portefeuilles simulés dont la composition exacte était
connue, puis appliqué aux vrais fonds ASFIM. L'inférence est instantanée une fois le modèle entraîné.

#### Comment lire les indicateurs

- **R² de réplication** — la part de la performance du fonds expliquée par sa répartition sectorielle
  estimée. Au-dessus de 0,70, la réplication est considérée fiable ; en-dessous, le fonds suit
  probablement une stratégie qui s'écarte des grands secteurs cotés (stock-picking marqué, petites
  valeurs, etc.).
- **Alpha net** — le rendement du fonds qui n'est PAS expliqué par ses expositions sectorielles :
  frais de gestion, sélection de titres, timing.
- **Tracking error** — l'ampleur des écarts semaine après semaine entre le fonds et sa réplication.
- **Poids actions moyen** — doit rester ≥ 60 % en permanence pour un fonds classé « Actions »
  (circulaire AMMC 02-09).
- **Dérive de style** — signale les semaines où la répartition sectorielle du fonds s'écarte
  fortement de sa moyenne habituelle.

#### Limites à connaître

- Le modèle a été entraîné sur environ 800 portefeuilles simulés. La méthode classique (QP,
  optimisation directe) reste actuellement plus précise sur certains critères ; passer à l'échelle
  recommandée (20 000+ portefeuilles) demande un réentraînement plus long mais devrait resserrer l'écart.
- Les secteurs marocains sont fortement corrélés entre eux (ex. Banques/Assurances/Holdings) :
  quand deux répartitions différentes expliquent presque aussi bien la même performance, le modèle
  ne peut pas toujours trancher avec certitude laquelle est la bonne — c'est une limite connue de
  toute méthode RBSA, pas un défaut de ce modèle en particulier.

#### Actualiser les données

Pour intégrer de nouvelles semaines de VL ASFIM ou des indices sectoriels mis à jour, dépose les
nouveaux fichiers dans `data/`, supprime le cache `outputs/dataset/asfim_ACTIONS_returns.parquet`,
puis relance `python scripts/run_pipeline.py`. Le détail complet est dans `README.md`.
    """)
