import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# --- CONFIGURATION ---
st.set_page_config(page_title="Analyse de survie des guerres commerciales", layout="wide")
st.title("📘 Analyse de survie des guerres commerciales : Modèle de Cox")

# --- ABSTRACT ---
st.header("🔍 Abstract")
st.markdown("""
Cette étude applique un modèle de **Cox Proportionnel de Risques** pour estimer la **durée probable des conflits commerciaux** bilatéraux.  
Les données proviennent de **Global Trade Alert** et sont enrichies avec les données du **PIB réel** par pays (World Bank).  
L’objectif est d’identifier les caractéristiques associées à la **persistance des tensions commerciales**, en modélisant leur durée en fonction de variables structurelles.

Cette approche permet de mieux anticiper la longévité des mesures protectionnistes et d’évaluer leur trajectoire probable dans le temps.  
Les résultats sont particulièrement utiles pour les **gérants de portefeuille**, qui peuvent ainsi mieux calibrer leurs allocations sectorielles et géographiques face aux risques géopolitiques et commerciaux.
""")


# --- INTRODUCTION ---
st.header("📘 Introduction")
st.markdown("""
Depuis 2018, les tensions commerciales entre grandes puissances se sont multipliées, mettant en péril la stabilité des échanges mondiaux.  
Des conflits comme celui entre les **États-Unis et la Chine** ont montré à quel point ces différends peuvent s’enliser dans le temps.

Pour comprendre **ce qui prolonge ou accélère la fin** de ces guerres commerciales, cette étude mobilise une approche issue de la médecine et des sciences de la survie :  
le **modèle de Cox proportionnel de risques**.

---

🔬 **Origines du modèle de Cox**  
Proposé en **1972** par le statisticien britannique **David Cox**, ce modèle a été conçu à l’origine pour analyser le **temps jusqu’à la survenue d’un événement**, comme un décès ou une rechute en épidémiologie.

Il s’est rapidement imposé dans d'autres domaines comme l’ingénierie (temps avant une panne), la finance (durée de vie d’un produit), ou encore la sociologie (temps jusqu’à un divorce, changement d’emploi, etc.).

---

📈 **Pourquoi ce choix méthodologique ?**  
Le modèle de Cox est bien adapté à la problématique de cette étude car il permet de :

- **Modéliser la durée d’un conflit** commercial en fonction de variables explicatives économiques ou politiques ;
- **Gérer des données censurées**, c’est-à-dire des conflits encore actifs au moment de l’analyse ;
- **Éviter d’imposer une forme rigide à la distribution des durées**, ce qui est souvent le cas dans d'autres méthodes.

---

🧩 **Interprétation attendue**  
Ce modèle permet d’estimer à chaque instant la **probabilité qu’un conflit se termine**, compte tenu de ses caractéristiques.  
Il devient ainsi possible d’identifier les **facteurs de persistance** ou au contraire les **éléments favorisant une résolution rapide**, avec des implications concrètes pour la diplomatie ou la planification économique.
""")

# --- METHODOLOGIE ---
st.header("🧪 Méthodologie")

st.markdown("""
Le modèle de **Cox Proportionnel de Risques** (Cox, 1972) permet d’estimer la probabilité qu’un **événement** (ici, la fin d’un conflit commercial) survienne à chaque instant, en fonction de certaines caractéristiques économiques et politiques.

Sa spécificité est de ne pas supposer de forme particulière pour le risque de base. Il permet donc de modéliser des **durées** sans faire d’hypothèse forte sur leur distribution.
""")

st.markdown("La fonction de risque estimée prend la forme :")
st.latex(r"h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k)")

st.markdown("où :")
st.markdown("- $h(t|X)$ est le **risque** qu’un conflit se termine à l’instant $t$ donné ses caractéristiques")
st.markdown("- $h_0(t)$ est le **risque de base** (identique pour tous les conflits)")
st.markdown("- Les $\\beta_k$ mesurent l’influence des variables explicatives")

st.markdown("Les variables utilisées ici sont :")
st.markdown("- $X_1$ = Croissance du PIB du pays initiateur (`gdp_country_a`)")
st.markdown("- $X_2$ = Nombre total de mesures imposées (`number_of_measures`)")
st.markdown("- $X_3$ = Durée moyenne des mesures (`mean_duration_months`)")


# --- DONNEES REELLES ---
data = pd.read_csv(r"df_episodes.csv")

# --- MODELE DE COX ---
cph = CoxPHFitter()

# ✅ Variables explicatives uniquement
features = [
    'gdp_country_a',
    'number_of_measures',
    'mean_duration_months'
]

# On garde aussi country_a, country_b pour filtrer ensuite
data_model = data[[
    'country_a', 'country_b'
] + features + ['duration_months', 'terminated']].dropna().copy()

# ✅ Ajouter un label lisible pour les graphiques
data_model['label'] = data_model['country_a'] + " – " + data_model['country_b']


# ✅ Corrigé : on garde uniquement les colonnes numériques pour le modèle
model_cols = ['gdp_country_a', 'number_of_measures', 'mean_duration_months', 'duration_months', 'terminated']
cph.fit(data_model[model_cols], duration_col='duration_months', event_col='terminated')


# --- RESULTATS ---
st.header("📈 Résultats")
st.subheader("Table des coefficients")
st.dataframe(cph.summary[['coef', 'exp(coef)', 'p']].round(3))



# --- COURBES DE SURVIE : conflits terminés initiés par grandes puissances ---
st.subheader("📈 Courbes de survie – conflits terminés (initiés par grandes puissances)")

big_players = ["United States of America", "India", "France", "United Kingdom", "Russia", "Canada", "Japan"]

# Filtrer conflits terminés
terminated_conflicts = data_model[data_model['terminated'] == 1].copy()

# Ajouter start_year
terminated_conflicts['start_year'] = pd.to_datetime(data.loc[terminated_conflicts.index, 'start_date']).dt.year

# Créer label
terminated_conflicts['label'] = (
    terminated_conflicts['country_a'] + " – " +
    terminated_conflicts['country_b'] + " (" +
    terminated_conflicts['start_year'].astype(str) + ")"
)

# 1. Sélectionner au max 5 conflits USA → big players
usa_terminated = (
    terminated_conflicts[
        (terminated_conflicts['country_a'] == "United States of America") &
        (terminated_conflicts['country_b'].isin(big_players))
    ]
    .sort_values('start_year')
    .head(5)
)

# 2. Compléter avec autres conflits big_players → big_players
remaining_needed = max(0, 10 - len(usa_terminated))

other_bigpower_conflicts = terminated_conflicts[
    (terminated_conflicts['country_a'].isin(big_players)) &
    (terminated_conflicts['country_a'] != "United States of America") &
    (terminated_conflicts['country_b'].isin(big_players)) &
    (~terminated_conflicts.index.isin(usa_terminated.index))
].groupby('country_a', group_keys=False).apply(lambda group: group.head(2)).reset_index(drop=True)

# Sélectionner uniquement ce qu'il faut
additional = other_bigpower_conflicts.head(remaining_needed)

# Concaténer
final_set = pd.concat([usa_terminated, additional]).reset_index(drop=True)

# Graphe
if len(final_set) > 0:
    surv_plot = cph.predict_survival_function(final_set.drop(columns='label'))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(final_set)):
        label = final_set['label'].iloc[i]
        ax.plot(surv_plot.index, surv_plot.iloc[:, i], label=label)

    ax.set_title("Probabilité de survie – conflits terminés (initiés par grandes puissances)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Probabilité que le conflit soit encore actif")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small", loc="best")
    st.pyplot(fig)
else:
    st.markdown("⚠️ Aucun conflit terminé détecté parmi les grandes puissances.")



# --- INSPECTION DÉTAILLÉE D'UN CONFLIT BILATÉRAL ---
st.subheader("🔎 Détails des conflits entre deux pays (filtrables)")
st.markdown("""
💡 À noter : lorsqu’un conflit est toujours en cours dans les données, la **probabilité de survie estimée reste proche de 100 %** tout au long de l’horizon considéré.  
Cela reflète simplement le fait que **le modèle ne dispose d’aucune information sur sa fin**, et n’a donc aucune base pour estimer un risque de sortie dans le futur proche.  
Ce phénomène est fréquent en analyse de survie dès lors qu’il y a des observations dites **censurées à droite** (c’est-à-dire non terminées à la date d’analyse).
""")


# Liste des pays initiateurs disponibles
available_initiators = sorted(data['country_a'].unique())
selected_initiator = st.selectbox("Choisissez un pays initiateur :", available_initiators, key="initiator_select")

# Liste des pays cibles disponibles pour ce pays
available_targets = sorted(data[data['country_a'] == selected_initiator]['country_b'].unique())
selected_target = st.selectbox("Choisissez un pays cible :", available_targets, key="target_select")

# Filtrer les conflits entre ces deux pays
filtered_conflicts = data[
    (data['country_a'] == selected_initiator) & 
    (data['country_b'] == selected_target)
].copy()

# Afficher tous les conflits entre les deux pays
st.dataframe(filtered_conflicts[[
    'country_a', 'country_b', 'start_year', 'start_date', 'end_date',
    'terminated', 'duration_months', 'number_of_measures',
    'mean_duration_months', 'gdp_country_a'
]])

# Sélectionner l'année du conflit à afficher
available_years = filtered_conflicts['start_year'].dropna().unique()
available_years.sort()
selected_year = st.selectbox("Choisissez l'année de début du conflit :", available_years, key="year_select")

# --- COURBE DE SURVIE POUR LE CONFLIT SÉLECTIONNÉ ---
st.subheader("📉 Courbe de survie pour ce conflit sélectionné")

required_cols = [
    'gdp_country_a', 'number_of_measures', 'mean_duration_months',
    'duration_months', 'terminated'
]

# Vérifier qu'au moins une ligne a toutes les colonnes nécessaires
valid_conflicts = filtered_conflicts.dropna(subset=required_cols).copy()

# Ajouter colonne start_year si absente
if 'start_year' not in valid_conflicts.columns:
    valid_conflicts['start_year'] = pd.to_datetime(valid_conflicts['start_date']).dt.year

# Filtrer sur l’année choisie
conflict_row = valid_conflicts[valid_conflicts['start_year'] == selected_year]

if len(conflict_row) > 0:
    # Créer un label plus informatif
    label_text = (
        conflict_row['country_a'].iloc[0]
        + " – "
        + conflict_row['country_b'].iloc[0]
        + " (" + str(conflict_row['start_year'].iloc[0]) + ")"
    )

    # Prédire la courbe de survie
    surv_pred = cph.predict_survival_function(conflict_row[required_cols])

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(surv_pred.index, surv_pred.iloc[:, 0], label=label_text, color='darkblue')
    ax3.set_title(f"Survie estimée : {label_text}")
    ax3.set_xlabel("Mois")
    ax3.set_ylabel("Probabilité que le conflit soit encore actif")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)
else:
    st.markdown("⚠️ Aucune donnée disponible pour cette année de conflit.")









# --- PREDICTION PERSONNALISÉE ---  
st.subheader("🔮 Prédiction personnalisée : USA – Chine (2025)")

# 👉 Entrées utilisateur via sliders
gdp_input = st.slider("Croissance du PIB du pays initiateur (%)", min_value=-10.0, max_value=10.0, value=2.3, step=0.1)
measures_input = st.slider("Nombre de mesures tarifaires imposées", min_value=1, max_value=50, value=15, step=1)
duration_input = st.slider("Durée moyenne des mesures (en mois)", min_value=1, max_value=36, value=8, step=1)

# 🧮 Créer l'input pour le modèle
custom_input = pd.DataFrame([{
    'gdp_country_a': gdp_input,
    'number_of_measures': measures_input,
    'mean_duration_months': duration_input
}])

# 🔮 Prédiction de la courbe de survie
surv_pred = cph.predict_survival_function(custom_input)

# 📉 Tracer la courbe avec axes dynamiques
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(surv_pred.index, surv_pred.iloc[:, 0], label="USA – Chine (2025)", color="red")
ax2.set_title("Survie prédite pour un conflit type USA – Chine 2025")
ax2.set_xlabel("Mois")
ax2.set_ylabel("Probabilité que le conflit soit encore actif")
ax2.set_ylim(surv_pred.min().min() - 0.05, min(1.05, surv_pred.max().max() + 0.05))
ax2.set_xlim(0, surv_pred.index.max())
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)



st.markdown("""  
### ℹ️ **Comment interpréter cette prédiction ?**

Le conflit USA–Chine (2025) est **déjà en cours**, mais sa **fin est inconnue**. Le modèle de survie permet de :

- **Estimer la probabilité qu’il soit encore actif** à chaque mois à venir,
- En se basant sur **des conflits passés similaires** (même type de mesures, durée moyenne, PIB...).
- Comparer l'impact des différentes variables.
            
---

##### Pourquoi la courbe est-elle utile ?
- Elle permet de **raisonner sur la persistance probable** du conflit, même si on ne connaît pas encore sa date de fin.
- Le modèle apprend des **conflits similaires passés**, y compris ceux encore actifs, pour estimer le **risque de fin** mois par mois.

---

##### Que signifie la courbe ?
- L’axe horizontal montre le **temps écoulé** depuis le début (en mois),
- L’axe vertical montre la **probabilité que le conflit soit encore actif** à ce moment-là.

---

👉 **Exemple** : si la courbe indique 80 % à 12 mois, cela signifie qu’il y a **80 % de chances que le conflit soit encore en cours** un an après son début, selon les données historiques.

---

Ce type de modèle permet donc d’**anticiper la persistance** des conflits, **même si leur issue reste inconnue à ce jour**.
""")



# --- INTERPRÉTATION DES RÉSULTATS ---
st.header("📊 Interprétation")
st.markdown("""
Les résultats du modèle de Cox estiment l’effet de certaines variables économiques sur la probabilité qu’un conflit commercial se termine à un instant donné. Un **coefficient négatif** signifie que la variable **réduit cette probabilité**, autrement dit, elle **prolonge le conflit**. À l’inverse, un coefficient positif indiquerait une accélération de la résolution.

Le **hazard ratio** (`exp(coef)`) représente l'effet multiplicatif de chaque variable sur le **taux de sortie du conflit**. Une valeur inférieure à 1 indique une **diminution du risque de fin**, donc une plus grande **persistance** du conflit.

---

##### 📌 1. Croissance du PIB du pays initiateur (`gdp_country_a`)
- Coefficient estimé : **–0.034**
- Hazard ratio : **0.967**
- Très significatif (**p ≈ 0**)

**Interprétation** : une croissance économique plus forte du pays initiateur **réduit légèrement** la probabilité que le conflit se termine à court terme. Le pays peut se permettre de maintenir ses positions, absorber les effets négatifs du conflit, et rester engagé plus longtemps.

> Le hazard ratio de 0.967 indique qu’à croissance plus élevée, la sortie du conflit devient **environ 3% moins probable** à chaque mois.

---

##### 📌 2. Nombre total de mesures tarifaires (`number_of_measures`)
- Coefficient estimé : **–0.544**
- Hazard ratio : **0.580**
- Très significatif (**p ≈ 0**)

**Interprétation** : plus le conflit est chargé de mesures tarifaires, plus il a tendance à **s’enliser**. Cela reflète une dynamique d’escalade, où chaque nouvelle barrière rend la négociation plus complexe.

> Le hazard ratio de 0.58 implique que chaque mesure supplémentaire **réduit de 42% la probabilité de résolution à un moment donné**.

---

##### 📌 3. Durée moyenne des mesures (`mean_duration_months`)
- Coefficient estimé : **–0.127**
- Hazard ratio : **0.881**
- Très significatif (**p ≈ 0**)

**Interprétation** : des mesures structurellement longues indiquent une **volonté politique de maintenir la pression**, ce qui ralentit mécaniquement toute issue négociée.

> Le hazard ratio de 0.881 implique qu’une durée moyenne plus longue des mesures **ralentit sensiblement la désescalade**.

---

##### Lecture globale des courbes de survie

Les conflits commerciaux, une fois enclenchés, ont **une forte probabilité de persister dans le temps**, parfois bien au-delà de 12 à 24 mois. Les cas de résolution rapide sont rares et souvent liés à des situations particulières à faible intensité.

""")



# --- CONCLUSION POUR LA GESTION DE PORTEFEUILLE ---
st.header("Conclusion et regard d'ensemble")

st.markdown("""
##### 📌 Ce que montre cette étude

Les tensions commerciales ne sont pas simplement des événements ponctuels ou réversibles. Lorsqu’elles démarrent avec une certaine intensité — en nombre ou en durée de mesures — elles tendent à **s’ancrer durablement dans le paysage économique mondial**.

Le modèle de Cox met en évidence trois leviers structurels de cette persistance :

- Une **croissance économique solide** permet aux pays initiateurs de maintenir leurs politiques protectionnistes.
- Un **nombre élevé de mesures** traduit une logique d’escalade difficile à inverser.
- Des **mesures longues** incarnent une stratégie d’endurance commerciale ou politique.

---

### Portfolio Management Point of View

En tant que professionnel de l’allocation d’actifs à long terme, ce type de modèle est **précieux pour ajuster les attentes macroéconomiques**, à défaut de prévoir des retournements politiques.

##### Voici comment cette analyse peut être utilisée concrètement :

- 📉 **Affiner les scénarios macroéconomiques** en intégrant une forte probabilité de persistance des tensions commerciales — notamment en lien avec le niveau de croissance du PIB du pays initiateur, le nombre de mesures tarifaires, et leur durée.
- 🧭 **Fournir un cadre analytique robuste** pour anticiper la temporalité des conflits — particulièrement utile dans les approches top-down de construction de portefeuille.
- 💬 **Informer et rassurer les clients institutionnels ou privés** avec des éléments empiriques sur les risques de prolongement des conflits, afin d’éviter des ajustements précipités face à des annonces politiques ponctuelles.
- 🔍 **Surveiller les points de rupture potentiels**, tels qu’une levée partielle de mesures tarifaires ou un changement politique, qui pourraient signaler un retournement dans la dynamique du conflit.

> *“Les tensions tarifaires ont un impact, mais notre analyse montre qu’elles suivent souvent une dynamique lente et prévisible. En gardant une vision long terme et structurée, on évite de sur-réagir aux à-coups politiques.”*            

---

### En synthèse

Ce modèle permet de comprendre que **certains conflits ne sont pas accidentels, mais structurels**. Il ne remplace pas une analyse politique, mais fournit une **boussole empirique** qui éclaire la dynamique temporelle des conflits commerciaux. Pour un gérant, cela signifie :

> Mieux cadrer l'incertitude. Mieux expliquer l’attente. Mieux protéger la vision long terme.

""")




# --- REFERENCES ---
st.header("📚 Références")
st.markdown("""
- Cox, D.R. (1972). Regression Models and Life-Tables. Journal of the Royal Statistical Society.
- Global Trade Alert: www.globaltradealert.org
- World Bank Open Data: https://data.worldbank.org
- Bown, C. P. (2020). US–China Trade War Tariffs: An Up-to-Date Chart. Peterson Institute.
""")


