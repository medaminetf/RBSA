# RBSA-ML — Phase 1 : Actions & Fonds Actions

Adaptation Machine Learning de l'analyse de style de Sharpe (RBSA) au marché
marocain — Africapital Management. **Branché sur données réelles** (secteurs
MASI 2013-2026, VL hebdo ASFIM fonds Actions).

**Principe** : au lieu de résoudre le problème inverse par QP à chaque fenêtre,
on génère massivement des portefeuilles synthétiques dont les poids θ* sont
connus, puis on entraîne un modèle supervisé (tête softmax = contraintes de
Sharpe) à retrouver ces poids depuis les seuls rendements. La fonction inverse
est apprise une fois pour toutes, contraintes incluses ; l'inférence est
instantanée.

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

**Le plus simple (Windows) :** double-clique sur `Lancer_application.bat` à la racine du
projet. Il installe les dépendances si besoin, entraîne le modèle au premier lancement
(~15 min, une seule fois), puis ouvre l'application dans le navigateur.

**En ligne de commande :**
```bash
python scripts/run_pipeline.py --quick   # démo (~3 min CPU)
python scripts/run_pipeline.py           # complet (~15 min CPU avec les données réelles)
streamlit run app/streamlit_app.py       # application interactive (navy/gold, fonds ASFIM réels)
```

L'application propose 4 onglets : **Vue d'ensemble** (panorama de tous les fonds Actions
ASFIM, triable), **Fiche du fonds** (détail d'un fonds : KPIs, alertes, dérive de style),
**Comparer des fonds** (jusqu'à 4 fonds côte à côte), **Comment ça marche** (explication
en langage courant des indicateurs et des limites du modèle).

Le pipeline enchaîne : **A** espace de style (réel, cf. ci-dessous) → **B**
génération du dataset synthétique (θ* connu, sur rendements sectoriels réels)
→ **C** entraînement MLP puis GRU → **D** baseline QP (CVXPY/OSQP) →
évaluation comparative → **E** application aux VL réelles ASFIM (fonds Actions).

Résultats dans `outputs/` : `evaluation_summary.csv`, `evaluation_detail.csv`,
`opcvm_reports.csv` (fonds réels analysés), `manifest.json`, `runs.csv` (log
d'entraînement), `models/` (checkpoints), `dataset/` (parquets + meta.json +
cache ASFIM consolidé).

## Données réelles branchées

Le dossier `data/` contient maintenant :

```
data/
├── masi/<SECTEUR>/<TITRE>.csv     # 177 fichiers, historiques Investing.com par titre
│   ├── _index.csv                 # indice sectoriel précalculé (équipondéré, base 1000)
│   └── _index_cap.csv             # variante cap-weighted
├── masi_indices/
│   ├── sector_indices.csv         # 24 indices sectoriels combinés (source PRÉFÉRÉE de X)
│   └── sector_indices_cap.csv     # variante cap-weighted
└── asfim_hebdomadaire/*.xlsx      # 633 instantanés hebdo ASFIM (2013-2026), toutes classifications
```

`config.yaml → style_space` pilote la source, dans cet ordre de préférence :
1. `precomputed_index_file` (indices déjà nettoyés — stale pricing, ticks
   aberrants, renouvellement de l'univers coté déjà gérés en amont) ;
2. `real_data_dir` (reconstruction depuis les titres bruts, en repli) ;
3. simulateur synthétique (si aucune donnée réelle n'est présente).

Les 24 secteurs de `style_space.sectors` correspondent **exactement** aux noms
de dossiers/colonnes réels (aucun mapping fragile). L'Étape E charge et met en
cache (`outputs/dataset/asfim_ACTIONS_returns.parquet`) les fonds classés
« ACTIONS » extraits des 633 fichiers hebdomadaires ASFIM ; les fonds
disparus/fusionnés sont conservés (pas de biais de survie).

## Bugs trouvés et corrigés pendant l'intégration réelle

Deux bugs sérieux sont apparus uniquement au contact des vraies données (invisibles
sur le simulateur synthétique, qui ne pouvait pas les révéler) :

1. **Parsing de dates (`dayfirst=True` aveugle)** — appliqué à des dates ISO
   (`2026-06-12`) ou au format Investing.com (`MM/DD/YYYY`), `dayfirst=True`
   permute jour et mois dès que les deux sont ≤ 12 (`2026-06-12` lu comme le
   6 décembre). Sur `sector_indices.csv`, ~60 % des dates étaient corrompues,
   ce qui décalait silencieusement toute la matrice de style réelle — la QP
   glissante n'expliquait alors que 4 % de la variance des vrais fonds ASFIM
   (R² ≈ 0,04) au lieu de 85-90 % attendus. Corrigé par un parseur qui détecte
   le format (`rbsa/data/loaders.py::_smart_parse_dates`) plutôt que de forcer
   un ordre jour/mois.
2. **Sélection de colonne de rendement** — sur les CSV Investing.com
   (`Date, Price, Open, High, Low, Vol., Change %`), l'ancienne heuristique
   (colonne la plus « numérique ») pouvait choisir `Price` (un niveau) au lieu
   de `Change %` (la vraie variation). Corrigé en priorisant les en-têtes
   explicites (`Change`, `Var.`, `%`...).

Une fois ces deux bugs corrigés, la baseline QP explique 83-93 % de la
variance des vrais fonds ASFIM testés — cohérent avec l'objectif documenté
par l'outillage existant du dossier `masi` (R² OLS sectoriel cible 0,60-0,90).
Un troisième ajustement (sans bug, calibration) : le bruit idiosyncratique du
générateur (Étape B) a été recalé sur le résidu réel observé (QP 52 sem. vs
VL ASFIM, médian ≈ 0,49 %/semaine) plutôt que sur une estimation a priori.

## Résultats (dernier run complet, données réelles)

| Modèle | MAE poids | Erreur rotation | R² réplication | Retard détection (sem.) |
|---|---|---|---|---|
| GRU | 0,039 | 0,006 | 0,75 | 44 |
| MLP | 0,038 | 0,009 | 0,63 | 50 |
| QP  | 0,027 | 0,005 | 0,91 | 24 |

Sur ce test synthétique (généré à partir des rendements sectoriels réels), la
**QP reste la référence la plus précise** à cette échelle d'entraînement
(800 portefeuilles). Sur les **vrais fonds ASFIM Actions**, le GRU produit des
diagnostics sensés (R² 0,83-0,90, tracking error 4-5,4 %, alpha nettement de
frais proche de zéro) — l'inférence sur données réelles fonctionne, mais
l'avantage du ML sur la QP annoncé par la spec (dynamique/rotation) n'est pas
encore net à cette échelle. La spec originale recommandait 20 000-50 000
portefeuilles synthétiques (`n_portfolios` dans `config.yaml`) contre 800 ici :
c'est le levier principal pour dépasser la QP sur la détection de rotation —
augmenter `n_portfolios` et relancer `run_pipeline.py` (~1-2h CPU à cette échelle).

## Brancher/actualiser les données réelles

1. **Rafraîchir les indices sectoriels** — remplacer `data/masi_indices/sector_indices.csv`
   par une version à jour (même format : `date` + une colonne par secteur, niveaux base 1000).
2. **Rafraîchir les VL ASFIM** — ajouter les nouveaux `.xlsx` hebdomadaires dans
   `data/asfim_hebdomadaire/`, puis supprimer le cache
   `outputs/dataset/asfim_ACTIONS_returns.parquet` (il n'est reconstruit que s'il est absent).
3. **MONIA réel** — remplacer l'approximation taux BAM/52 en passant la série
   hebdo à `cash_factor` (cf. `rbsa/data/sectors.py`).
4. **Cap-weighted** — `style_space.precomputed_index_file_cap` pointe déjà
   vers `sector_indices_cap.csv` ; basculer `precomputed_index_file` dessus.

## Structure

```
rbsa/
├── config.py                 # YAML, seed, matrice de passage 24→macro
├── data/
│   ├── loaders.py            # données réelles (titres, ASFIM, indices précalculés), parsing défensif
│   ├── sectors.py            # indices sectoriels, facteur CASH, agrégation
│   └── synthetic_market.py   # sélection réel/synthétique + winsorisation
├── generator/
│   ├── archetypes.py         # quasi-indiciel / diversifié / paris sectoriels / concentré
│   └── generator.py          # marche aléatoire simplexe, frais, bruit, rotations brutales
├── models/
│   ├── qp_baseline.py        # RBSA Sharpe : QP CVXPY/OSQP fenêtre glissante + Ridge
│   ├── networks.py           # InverseMLP, InverseGRU (softmax head)
│   └── losses.py             # L = L_poids (KL/MSE) + λ·L_reconstruction + λ_rot·L_rotation
├── training/
│   ├── dataset.py            # fenêtres/séquences, z-score PLANCHÉ + clippé (stats train only)
│   └── train.py              # early stopping, log CSV, inférence
├── evaluation/metrics.py     # MAE poids, rotation, retard de détection, R², alpha
└── application/opcvm.py      # conformité AMMC : ≥60% actions, R²≥0.70, dérive style
```

## Garde-fous méthodologiques

* **Fuite temporelle** : splits par période (2013-2020 / 2021-2022 / 2023-2026)
  ET par portefeuille ; standardisation calculée sur le train uniquement.
* **Contraintes de Sharpe par construction** : sortie softmax → θ ≥ 0, Σθ = 1.
* **Perte physique** : le terme de reconstruction ‖y − Xθ̂‖² garde le modèle
  cohérent avec l'objectif RBSA original.
* **Rendements arithmétiques simples** partout (additivité transversale).
* **Jamais de forward-fill** sur un titre : renormalisation sur l'univers coté.
* **Plancher d'écart-type + clipping en standardisation** : un secteur non
  encore coté durant tout le train (écart-type ~0) ne fait plus exploser les
  z-scores une fois qu'il redevient actif en validation/test.
* **Parsing de dates format-aware** (voir bugs ci-dessus) : jamais de
  `dayfirst` supposé sans détection du format source.
* **Reproductibilité** : seed global unique, config YAML archivée avec chaque
  dataset (`outputs/dataset/meta.json`).

## Extension phase 2 (interfaces prêtes)

* `K` est paramétrable partout : ajouter MBI CT/MT/MLT + MONIA dans
  `style_space.sectors` couvre les fonds obligataires/diversifiés sans réécriture.
* `build_features(add_lags=…)` : les retards `R_{k,t-1}, R_{k,t-2}` (stale
  pricing / lissage mark-to-model) sont déjà câblés, désactivés en phase 1.
* Le générateur accepte n'importe quel espace de facteurs (mêmes archétypes,
  contrainte de sensibilité à ajouter dans `archetypes.py`).
