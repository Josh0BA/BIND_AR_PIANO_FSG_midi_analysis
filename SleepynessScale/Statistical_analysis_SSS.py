import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, t, ttest_rel, wilcoxon

try:
	from statsmodels.stats.anova import AnovaRM
except ImportError as exc:
	raise ImportError(
		"statsmodels is required for repeated-measures ANOVA. Install with: pip install statsmodels"
	) from exc


PHASES = ["Pretest", "Lernphase", "Posttest"]


def load_sss_data(script_dir: str) -> pd.DataFrame:
	excel_path = os.path.join(script_dir, "SleepinessScale_TN.xlsx")
	csv_path = os.path.join(script_dir, "SleepinessScale_TN.csv")

	if os.path.exists(excel_path):
		try:
			return pd.read_excel(excel_path, engine="openpyxl")
		except Exception as excel_error:
			try:
				print("Hinweis: XLSX via 'calamine' geladen (openpyxl konnte die Datei nicht lesen).")
				return pd.read_excel(excel_path, engine="calamine")
			except Exception as calamine_error:
				if os.path.exists(csv_path):
					print("Hinweis: XLSX konnte nicht gelesen werden. Nutze stattdessen CSV.")
					return pd.read_csv(csv_path)
				raise ValueError(
					"XLSX konnte weder mit openpyxl noch mit calamine gelesen werden. "
					f"openpyxl: {excel_error}; calamine: {calamine_error}. "
					f"Bitte als CSV exportieren nach: {csv_path}"
				)

	if os.path.exists(csv_path):
		return pd.read_csv(csv_path)

	raise FileNotFoundError(
		f"Keine Datei gefunden. Erwartet: {excel_path} oder {csv_path}"
	)


def prepare_dataframes(df_raw: pd.DataFrame):
	missing = [col for col in PHASES if col not in df_raw.columns]
	if missing:
		raise ValueError(f"Fehlende Spalten in Sleepiness-Daten: {missing}")

	if "Participant_ID" in df_raw.columns:
		participant_ids = df_raw["Participant_ID"].astype(str)
	else:
		participant_ids = pd.Series([f"P{i+1:03d}" for i in range(len(df_raw))], name="Participant_ID")

	df_wide = pd.DataFrame({"Participant_ID": participant_ids})
	for phase in PHASES:
		df_wide[phase] = pd.to_numeric(df_raw[phase], errors="coerce")

	df_wide_complete = df_wide.dropna(subset=PHASES).copy()
	if df_wide_complete.empty:
		raise ValueError("Keine vollständigen Fälle für Pretest/Lernphase/Posttest vorhanden.")

	df_long_complete = df_wide_complete.melt(
		id_vars="Participant_ID",
		value_vars=PHASES,
		var_name="Phase",
		value_name="Skala_Wert",
	)

	return df_wide, df_wide_complete, df_long_complete


def compute_descriptive(df_wide: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
	rows = []
	for phase in PHASES:
		values = pd.to_numeric(df_wide[phase], errors="coerce").dropna()
		n = len(values)
		if n == 0:
			continue

		mean_val = float(values.mean())
		std_val = float(values.std(ddof=1)) if n > 1 else np.nan
		median_val = float(values.median())
		min_val = float(values.min())
		max_val = float(values.max())
		se = std_val / np.sqrt(n) if n > 1 and not np.isnan(std_val) else np.nan

		if n > 1 and not np.isnan(se):
			t_crit = t.ppf(1 - alpha / 2, n - 1)
			ci_half = float(t_crit * se)
			ci_lower = mean_val - ci_half
			ci_upper = mean_val + ci_half
		else:
			ci_half = np.nan
			ci_lower = np.nan
			ci_upper = np.nan

		rows.append(
			{
				"Phase": phase,
				"N": n,
				"Mittelwert": mean_val,
				"Standardabweichung": std_val,
				"Median": median_val,
				"Minimum": min_val,
				"Maximum": max_val,
				"Standardfehler": se,
				"95%_KI_Halbbreite": ci_half,
				"95%_KI_unten": ci_lower,
				"95%_KI_oben": ci_upper,
			}
		)

	return pd.DataFrame(rows)


def compute_normality(df_wide_complete: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for phase in PHASES:
		values = df_wide_complete[phase].dropna()
		n = len(values)
		if n >= 3:
			stat, p_value = shapiro(values)
		else:
			stat, p_value = np.nan, np.nan

		rows.append(
			{
				"Phase": phase,
				"N": n,
				"Shapiro_W": stat,
				"Shapiro_p": p_value,
				"Normal_(p>=0.05)": bool(p_value >= 0.05) if not np.isnan(p_value) else False,
			}
		)

	return pd.DataFrame(rows)


def run_rm_anova(df_long_complete: pd.DataFrame) -> pd.DataFrame:
	model = AnovaRM(
		data=df_long_complete,
		depvar="Skala_Wert",
		subject="Participant_ID",
		within=["Phase"],
	)
	result = model.fit()
	table = result.anova_table.reset_index().rename(columns={"index": "Effekt"})
	return table


def run_friedman(df_wide_complete: pd.DataFrame) -> pd.DataFrame:
	arrays = [df_wide_complete[phase].to_numpy() for phase in PHASES]
	stat, p_value = friedmanchisquare(*arrays)
	return pd.DataFrame(
		[
			{
				"Test": "Friedman",
				"N_Subjekte": int(df_wide_complete["Participant_ID"].nunique()),
				"Bedingungen": len(PHASES),
				"Statistik": stat,
				"p_Wert": p_value,
				"Signifikant_(p<0.05)": bool(p_value < 0.05),
			}
		]
	)


def run_posthoc(df_wide_complete: pd.DataFrame, use_parametric: bool) -> pd.DataFrame:
	pairs = list(combinations(PHASES, 2))
	rows = []

	for left, right in pairs:
		left_values = df_wide_complete[left]
		right_values = df_wide_complete[right]
		diffs = left_values - right_values
		n_pairs = len(diffs)

		if use_parametric:
			stat, p_value = ttest_rel(left_values, right_values)
			if diffs.std(ddof=1) == 0:
				effect = np.nan
			else:
				effect = float(diffs.mean() / diffs.std(ddof=1))
			effect_name = "Cohens_dz"
			test_name = "paired_t_test"
		else:
			try:
				stat, p_value = wilcoxon(left_values, right_values, zero_method="wilcox")
			except ValueError:
				stat, p_value = np.nan, 1.0
			effect = np.nan
			effect_name = "Effektstaerke"
			test_name = "wilcoxon_signed_rank"

		rows.append(
			{
				"Vergleich": f"{left}_vs_{right}",
				"Test": test_name,
				"N_Paare": n_pairs,
				"Statistik": stat,
				"p_Wert": p_value,
				effect_name: effect,
			}
		)

	result = pd.DataFrame(rows)
	result["p_Wert_Bonferroni"] = np.minimum(result["p_Wert"] * len(pairs), 1.0)
	result["Signifikant_Bonferroni_(p<0.05)"] = result["p_Wert_Bonferroni"] < 0.05
	return result


def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))

	df_raw = load_sss_data(script_dir)
	df_wide, df_wide_complete, df_long_complete = prepare_dataframes(df_raw)

	descriptive_df = compute_descriptive(df_wide)
	normality_df = compute_normality(df_wide_complete)

	all_normal = bool(normality_df["Normal_(p>=0.05)"].all())
	if all_normal:
		inferential_df = run_rm_anova(df_long_complete)
		selected_test = "RM-ANOVA (repeated measures, parametric)"
	else:
		inferential_df = run_friedman(df_wide_complete)
		selected_test = "Friedman (repeated measures, non-parametric)"

	posthoc_df = run_posthoc(df_wide_complete, use_parametric=all_normal)

	decision_df = pd.DataFrame(
		[
			{
				"Ausgewaehlter_Test": selected_test,
				"Regel": "Alle Shapiro p >= 0.05 -> RM-ANOVA, sonst Friedman",
				"N_vollstaendige_Subjekte": int(df_wide_complete["Participant_ID"].nunique()),
			}
		]
	)

	out_desc = os.path.join(script_dir, "sleepiness_scale_descriptive_statistics.csv")
	out_inferential = os.path.join(script_dir, "sleepiness_scale_inferential_results.csv")
	out_posthoc = os.path.join(script_dir, "sleepiness_scale_posthoc_results.csv")

	descriptive_df.to_csv(out_desc, index=False)
	inferential_df.to_csv(out_inferential, index=False)
	posthoc_df.to_csv(out_posthoc, index=False)

	print("Analyse abgeschlossen. Dateien gespeichert:")
	for path in [out_desc, out_inferential, out_posthoc]:
		print(f" - {path}")


if __name__ == "__main__":
	main()
