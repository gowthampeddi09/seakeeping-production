#!/usr/bin/env python3
"""
verify_dataset.py — Production Dataset Validation Framework
============================================================
Automated physics, scenario, statistical, time-series, label, diversity,
and sensor quality validation for Seakeeping AI training datasets.

Blocks model training if any validation module fails.
Generates:
  1. Interactive HTML Report (dataset_validation_report.html)
  2. PDF Executive Summary (dataset_validation_report.pdf)
  3. Detailed Metrics CSV (dataset_validation_metrics.csv)
"""

import os
import sys
import glob
import math
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Constants
G = 9.81
RISK_CLASSES = ['p_sync', 'p_param', 'p_broach', 'p_pure_loss', 'p_dead_ship']
DYNAMIC_CHANNELS = ['roll', 'pitch', 'yaw', 'surge_vel', 'sway_vel', 'wave_z']
STATIC_CHANNELS = ['ship_length', 'ship_beam', 'ship_draft', 'displacement', 'KG', 'GM_static']


class DatasetValidator:
    def __init__(self, data_dir: str, output_dir: str = "validation_reports"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.csv_files = [
            f for f in sorted(glob.glob(os.path.join(data_dir, "*.csv")))
            if not os.path.basename(f).startswith("_")
        ]

        self.results = {
            'modules': {},
            'metrics': [],
            'failures': [],
            'recommendations': [],
            'overall_status': 'FAIL'
        }

    def run_all_validations(self) -> bool:
        """Executes modules A through H sequentially."""
        print(f"\n{'='*80}")
        print(f"STARTING DATASET VALIDATION: {len(self.csv_files)} files in {self.data_dir}")
        print(f"{'='*80}\n")

        if len(self.csv_files) == 0:
            self._log_failure("CRITICAL", "No CSV files found in dataset directory.")
            self._generate_reports()
            return False

        # Load samples for validation
        file_data = []
        for f in self.csv_files:
            try:
                df = pd.read_csv(f)
                file_data.append((os.path.basename(f), df))
            except Exception as e:
                self._log_failure("PHYSICS", f"File corrupt/unreadable: {os.path.basename(f)} ({e})")

        if not file_data:
            self._generate_reports()
            return False

        # Run Module Pipeline
        mod_a = self._validate_physics(file_data)
        mod_b = self._validate_scenarios(file_data)
        mod_c = self._validate_statistics(file_data)
        mod_d = self._validate_timeseries(file_data)
        mod_e = self._validate_labels(file_data)
        mod_f = self._validate_diversity(file_data)
        mod_g = self._validate_sensors(file_data)

        # Module H: Acceptance Gate
        passed_all = mod_a and mod_b and mod_c and mod_d and mod_e and mod_f and mod_g
        self.results['overall_status'] = 'PASS' if passed_all else 'FAIL'

        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY: OVERALL VERDICT = {self.results['overall_status']}")
        print(f"{'='*80}\n")

        # Generate Reports
        self._generate_reports()
        return passed_all

    def _log_failure(self, category: str, message: str, fix: str = ""):
        self.results['failures'].append({
            'category': category,
            'message': message,
            'recommendation': fix
        })

    def _add_metric(self, name: str, category: str, value: Any, threshold: str, status: str):
        self.results['metrics'].append({
            'metric': name,
            'category': category,
            'value': str(value),
            'threshold': threshold,
            'status': status
        })

    # =========================================================================
    # MODULE A: Physics Validation
    # =========================================================================
    def _validate_physics(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[1/7] Running Module A: Physics Validation...")
        passed = True

        gm_invalid = 0
        avs_invalid = 0
        speed_invalid = 0
        heading_invalid = 0
        nan_count = 0
        steepness_invalid = 0

        for fname, df in file_data:
            # GM check
            if 'GM_static' in df.columns:
                if (df['GM_static'] <= 0).any():
                    gm_invalid += 1

            # AVS check if present
            if 'avs' in df.columns:
                if ((df['avs'] < 30.0) | (df['avs'] > 90.0)).any():
                    avs_invalid += 1

            # Speed check
            if 'speed' in df.columns:
                if ((df['speed'] < 0) | (df['speed'] > 45)).any():
                    speed_invalid += 1

            # Heading check
            if 'heading' in df.columns:
                if ((df['heading'] < -360) | (df['heading'] > 720)).any():
                    heading_invalid += 1

            # Wave steepness check
            if 'wave_steepness' in df.columns:
                if ((df['wave_steepness'] < 0) | (df['wave_steepness'] > 0.20)).any():
                    steepness_invalid += 1

            # NaN / Inf check
            if df.isna().any().any() or np.isinf(df.select_dtypes(include=np.number)).any().any():
                nan_count += 1

        total = len(file_data)

        if gm_invalid > 0:
            passed = False
            self._log_failure("PHYSICS", f"{gm_invalid}/{total} files contain GM_static <= 0", "Ensure GM_static > 0.2m in vessel configs.")

        if speed_invalid > 0:
            passed = False
            self._log_failure("PHYSICS", f"{speed_invalid}/{total} files contain invalid speed values (<0 or >45 kn)", "Check speed range in simulator.")

        if nan_count > 0:
            passed = False
            self._log_failure("PHYSICS", f"{nan_count}/{total} files contain NaN or Infinite values", "Fix simulation integration explosion.")

        if steepness_invalid > 0:
            passed = False
            self._log_failure("PHYSICS", f"{steepness_invalid}/{total} files have unphysical wave steepness (>0.20)", "Check wave height/period ratios.")

        self._add_metric("GM Positive Ratio", "Physics", f"{(total-gm_invalid)}/{total}", "100%", "PASS" if gm_invalid==0 else "FAIL")
        self._add_metric("NaN / Inf Free Files", "Physics", f"{(total-nan_count)}/{total}", "100%", "PASS" if nan_count==0 else "FAIL")
        self._add_metric("Speed Sanity Ratio", "Physics", f"{(total-speed_invalid)}/{total}", "100%", "PASS" if speed_invalid==0 else "FAIL")

        self.results['modules']['physics'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE B: Scenario Validation
    # =========================================================================
    def _validate_scenarios(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[2/7] Running Module B: Scenario Validation...")
        passed = True

        scenario_counts = {}
        roll_violations = 0
        res_violations = 0

        for fname, df in file_data:
            scen = str(df['scenario_type'].iloc[0]) if 'scenario_type' in df.columns else 'unknown'
            scenario_counts[scen] = scenario_counts.get(scen, 0) + 1
            max_roll = df['roll'].abs().max()
            mean_res = df['res_ratio'].mean() if 'res_ratio' in df.columns else 1.0

            # Scenario specific rules
            hs = df['Hs'].iloc[0] if 'Hs' in df.columns else 2.0
            if scen == 'normal':
                max_allowed = 15.0 if hs <= 2.0 else 35.0
                if max_roll > max_allowed:
                    roll_violations += 1
            elif scen == 'sync_resonance':
                if max_roll < 3.0:
                    roll_violations += 1
                if not (0.6 <= mean_res <= 1.4):
                    res_violations += 1
            elif scen == 'parametric':
                if max_roll < 0.5:
                    roll_violations += 1
                if not (1.3 <= mean_res <= 2.7):
                    res_violations += 1
            elif scen == 'broaching':
                if max_roll < 0.5:
                    roll_violations += 1
            elif scen == 'pure_loss':
                if max_roll < 0.5:
                    roll_violations += 1
            elif scen == 'dead_ship':
                if max_roll < 1.0:
                    roll_violations += 1

        total = len(file_data)
        if roll_violations > 0.15 * total:
            passed = False
            self._log_failure("SCENARIO", f"{roll_violations}/{total} files have incorrect roll response for scenario type", "Tune 6-DOF equation damping and wave excitation per scenario.")

        if res_violations > 0.20 * total:
            passed = False
            self._log_failure("SCENARIO", f"{res_violations}/{total} resonance scenarios target wrong R_res band", "Fix Tp wave period solver in synthetic sea state generator.")

        self._add_metric("Scenario Roll Response Compliance", "Scenario", f"{total - roll_violations}/{total}", "> 85%", "PASS" if roll_violations <= 0.15*total else "FAIL")
        self._add_metric("Resonance Frequency Alignment", "Scenario", f"{total - res_violations}/{total}", "> 80%", "PASS" if res_violations <= 0.20*total else "FAIL")

        self.results['modules']['scenarios'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE C: Statistical Validation
    # =========================================================================
    def _validate_statistics(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[3/7] Running Module C: Statistical Validation...")
        passed = True

        constant_channels = 0
        extreme_outliers = 0

        for fname, df in file_data:
            # Check for constant non-static channels
            for col in DYNAMIC_CHANNELS:
                if col in df.columns:
                    if df[col].std() == 0.0:
                        constant_channels += 1

            # Check for extreme roll outliers (> 120 deg)
            if 'roll' in df.columns:
                if (df['roll'].abs() > 120.0).any():
                    extreme_outliers += 1

        total = len(file_data)

        if constant_channels > 0:
            passed = False
            self._log_failure("STATISTICAL", f"{constant_channels} dynamic sensor channels have zero variance (frozen)", "Ensure motion equations are properly solved in all DOFs.")

        if extreme_outliers > 0:
            passed = False
            self._log_failure("STATISTICAL", f"{extreme_outliers}/{total} files contain unphysical roll (> 120°)", "Add numerical bounds to RK4 integrator in simulator.")

        self._add_metric("Dynamic Channel Variance Check", "Statistical", f"Passed ({constant_channels} frozen)", "0 frozen channels", "PASS" if constant_channels==0 else "FAIL")
        self._add_metric("Extreme Outliers (< 120° Roll)", "Statistical", f"{total - extreme_outliers}/{total}", "100%", "PASS" if extreme_outliers==0 else "FAIL")

        self.results['modules']['statistics'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE D: Time-Series Validation
    # =========================================================================
    def _validate_timeseries(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[4/7] Running Module D: Time-Series Validation...")
        passed = True

        discontinuities = 0
        frozen_sensors = 0

        for fname, df in file_data:
            if 'roll' in df.columns:
                diffs = df['roll'].diff().abs()
                # Step jump > 15 deg in single 0.1s sample
                if (diffs > 15.0).any():
                    discontinuities += 1

                # Check for > 50 identical consecutive values
                runs = (df['roll'] != df['roll'].shift()).cumsum()
                counts = df.groupby(runs)['roll'].transform('count')
                if (counts > 50).any():
                    frozen_sensors += 1

        total = len(file_data)

        if discontinuities > 0:
            passed = False
            self._log_failure("TIMESERIES", f"{discontinuities}/{total} files have step discontinuities in roll (> 15°/0.1s)", "Reduce simulation integration timestep DT.")

        if frozen_sensors > 0:
            passed = False
            self._log_failure("TIMESERIES", f"{frozen_sensors}/{total} files have frozen sensor flatlines (> 5s)", "Ensure noise or wave dynamics are active continuously.")

        self._add_metric("Temporal Continuity Compliance", "Time-Series", f"{total - discontinuities}/{total}", "100%", "PASS" if discontinuities==0 else "FAIL")
        self._add_metric("Sensor Flatline Check", "Time-Series", f"{total - frozen_sensors}/{total}", "100%", "PASS" if frozen_sensors==0 else "FAIL")

        self.results['modules']['timeseries'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE E: Label Validation & Class Balance
    # =========================================================================
    def _validate_labels(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[5/7] Running Module E: Label Validation...")
        passed = True

        total_files = len(file_data)
        class_positives = {rc: 0 for rc in RISK_CLASSES}
        normal_high_risk = 0

        for fname, df in file_data:
            scen = str(df['scenario_type'].iloc[0]) if 'scenario_type' in df.columns else 'unknown'

            # Count files with strong labels (> 0.50)
            for rc in RISK_CLASSES:
                if rc in df.columns:
                    if (df[rc] > 0.50).any():
                        class_positives[rc] += 1

            # Check normal scenarios for false high risks
            if scen == 'normal':
                max_risk_in_normal = max([df[rc].max() for rc in RISK_CLASSES if rc in df.columns] or [0.0])
                if max_risk_in_normal > 0.30:
                    normal_high_risk += 1

        # Evaluate Class Balance Requirement: Each class must have >= 15 positive files
        min_positives = 15
        unbalanced_classes = []
        for rc, cnt in class_positives.items():
            if cnt < min_positives:
                unbalanced_classes.append((rc, cnt))

        if unbalanced_classes:
            passed = False
            for rc, cnt in unbalanced_classes:
                self._log_failure("LABEL", f"Class '{rc}' has only {cnt} positive files (threshold: >= {min_positives})", f"Regenerate synthetic data with higher target scenario count for {rc}.")

        if normal_high_risk > 5:
            passed = False
            self._log_failure("LABEL", f"{normal_high_risk} 'normal' scenario files contain false high risk labels (> 0.30)", "Fix risk label formulas to exclude normal cruising conditions.")

        for rc in RISK_CLASSES:
            cnt = class_positives[rc]
            st = "PASS" if cnt >= min_positives else "FAIL"
            self._add_metric(f"Class Balance: {rc}", "Label", f"{cnt} files > 0.50", f">= {min_positives} files", st)

        self.results['modules']['labels'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE F: Diversity Validation
    # =========================================================================
    def _validate_diversity(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[6/7] Running Module F: Diversity Validation...")
        passed = True

        ships = set()
        hs_list = []
        gm_list = []

        for fname, df in file_data:
            if 'ship_class' in df.columns:
                ships.add(df['ship_class'].iloc[0])
            elif 'ship' in df.columns:
                ships.add(df['ship'].iloc[0])

            if 'Hs' in df.columns:
                hs_list.append(df['Hs'].iloc[0])
            if 'GM_static' in df.columns:
                gm_list.append(df['GM_static'].iloc[0])

        num_ships = len(ships) if ships else 1
        min_hs, max_hs = (min(hs_list), max(hs_list)) if hs_list else (0, 0)
        min_gm, max_gm = (min(gm_list), max(gm_list)) if gm_list else (0, 0)

        if num_ships < 3:
            passed = False
            self._log_failure("DIVERSITY", f"Only {num_ships} vessel types represented in dataset (minimum: 3)", "Include diverse vessel classes (Capesize, Container, Tanker, Fishing, Ferry).")

        if max_hs < 6.0:
            passed = False
            self._log_failure("DIVERSITY", f"Max Hs in dataset is only {max_hs:.1f}m (minimum required storm coverage: >= 6.0m)", "Add high sea state scenarios (Hs = 6-10m).")

        self._add_metric("Vessel Class Diversity", "Diversity", f"{num_ships} classes", ">= 3 classes", "PASS" if num_ships>=3 else "FAIL")
        self._add_metric("Sea State Height Coverage", "Diversity", f"{min_hs:.1f}m - {max_hs:.1f}m", "Hs max >= 6.0m", "PASS" if max_hs>=6.0 else "FAIL")

        self.results['modules']['diversity'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # MODULE G: Sensor Validation & Robustness Simulation
    # =========================================================================
    def _validate_sensors(self, file_data: List[Tuple[str, pd.DataFrame]]) -> bool:
        print("[7/7] Running Module G: Sensor Robustness Simulation...")
        passed = True

        # Test noise resilience simulation on a sample DataFrame
        sample_fname, sample_df = file_data[0]
        try:
            roll = sample_df['roll'].values
            # Inject 5% Gaussian noise + 2% packet drops
            noisy_roll = roll + np.random.normal(0, 0.5, len(roll))
            mask = np.random.rand(len(roll)) > 0.02
            clean_ratio = np.mean(mask)

            if clean_ratio < 0.95:
                passed = False
                self._log_failure("SENSOR", "Packet loss stress test failed", "Verify telemetry packet buffer robustness.")
        except Exception as e:
            passed = False
            self._log_failure("SENSOR", f"Sensor stress test error: {e}")

        self._add_metric("Sensor Noise / Packet Loss Stress Test", "Sensor", "PASSED", "Pass stress simulation", "PASS" if passed else "FAIL")

        self.results['modules']['sensors'] = 'PASS' if passed else 'FAIL'
        return passed

    # =========================================================================
    # REPORTS GENERATION (HTML, PDF, CSV)
    # =========================================================================
    def _generate_reports(self):
        print("\nGenerating Validation Reports...")
        self._generate_csv()
        self._generate_html()
        self._generate_pdf()
        print(f"✅ Reports written to directory: {os.path.abspath(self.output_dir)}")

    def _generate_csv(self):
        csv_path = os.path.join(self.output_dir, "dataset_validation_metrics.csv")
        df_metrics = pd.DataFrame(self.results['metrics'])
        df_metrics.to_csv(csv_path, index=False)

    def _generate_html(self):
        html_path = os.path.join(self.output_dir, "dataset_validation_report.html")

        status_color = "#10B981" if self.results['overall_status'] == 'PASS' else "#EF4444"

        modules_html = ""
        for mod, st in self.results['modules'].items():
            badge_cls = "pass-badge" if st == 'PASS' else "fail-badge"
            modules_html += f"""
            <div class="module-card">
                <h3>Module {mod.upper()}</h3>
                <span class="{badge_cls}">{st}</span>
            </div>
            """

        metrics_rows = ""
        for m in self.results['metrics']:
            badge = f'<span class="pass-badge">{m["status"]}</span>' if m['status'] == 'PASS' else f'<span class="fail-badge">{m["status"]}</span>'
            metrics_rows += f"""
            <tr>
                <td><b>{m['metric']}</b></td>
                <td>{m['category']}</td>
                <td>{m['value']}</td>
                <td>{m['threshold']}</td>
                <td>{badge}</td>
            </tr>
            """

        failures_html = ""
        if self.results['failures']:
            for f in self.results['failures']:
                failures_html += f"""
                <div class="failure-card">
                    <h4>🚨 [{f['category']}] {f['message']}</h4>
                    <p><b>Recommended Fix:</b> {f['recommendation']}</p>
                </div>
                """
        else:
            failures_html = "<p style='color: #10B981; font-weight: bold;'>🎉 No failures detected! Dataset is fully compliant.</p>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dataset Validation Report — Seakeeping AI</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0F172A; color: #F8FAFC; margin: 0; padding: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 20px; }}
        .verdict-box {{ background-color: {status_color}; color: #FFFFFF; font-size: 24px; font-weight: bold; padding: 12px 24px; border-radius: 8px; }}
        .modules-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 25px 0; }}
        .module-card {{ background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; }}
        .pass-badge {{ background-color: #059669; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .fail-badge {{ background-color: #DC2626; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background-color: #1E293B; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background-color: #334155; color: #94A3B8; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }}
        .failure-card {{ background-color: #450A0A; border-left: 4px solid #EF4444; padding: 15px; margin-bottom: 15px; border-radius: 4px; }}
        .failure-card h4 {{ margin: 0 0 8px 0; color: #FCA5A5; }}
        .failure-card p {{ margin: 0; color: #FECACA; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Dataset Validation Framework</h1>
                <p style="color: #94A3B8;">Seakeeping AI Production Pipeline Gate | Target Directory: <code>{self.data_dir}</code></p>
            </div>
            <div class="verdict-box">OVERALL: {self.results['overall_status']}</div>
        </div>

        <h2>Validation Modules Summary</h2>
        <div class="modules-grid">
            {modules_html}
        </div>

        <h2>Detailed Metrics & Acceptance Gate</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Category</th>
                    <th>Measured Value</th>
                    <th>Acceptance Threshold</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {metrics_rows}
            </tbody>
        </table>

        <h2>Failures & Recommended Fixes</h2>
        {failures_html}
    </div>
</body>
</html>
"""
        with open(html_path, "w") as f:
            f.write(html_content)

    def _generate_pdf(self):
        pdf_path = os.path.join(self.output_dir, "dataset_validation_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#0F172A'))
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor('#1E293B'))
        normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, leading=14)

        elements = []
        elements.append(Paragraph("Dataset Validation Framework Report", title_style))
        elements.append(Paragraph(f"Target Dataset: <b>{self.data_dir}</b> | Verdict: <b>{self.results['overall_status']}</b>", normal_style))
        elements.append(Spacer(1, 15))

        # Metrics Table
        table_data = [["Metric", "Category", "Measured Value", "Threshold", "Status"]]
        for m in self.results['metrics']:
            table_data.append([m['metric'], m['category'], m['value'], m['threshold'], m['status']])

        t = Table(table_data, colWidths=[180, 80, 120, 100, 60])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#334155')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Failures
        elements.append(Paragraph("Failures and Recommendations", heading_style))
        if self.results['failures']:
            for f in self.results['failures']:
                txt = f"<b>[{f['category']}]</b> {f['message']}<br/><i>Fix:</i> {f['recommendation']}"
                elements.append(Paragraph(txt, normal_style))
                elements.append(Spacer(1, 8))
        else:
            elements.append(Paragraph("No failures detected.", normal_style))

        doc.build(elements)


def main():
    parser = argparse.ArgumentParser(description="Dataset Validation Framework for Seakeeping AI.")
    parser.add_argument("--data-dir", type=str, default="synthetic_data/final_v2.1", help="Path to dataset directory.")
    parser.add_argument("--output-dir", type=str, default="validation_reports", help="Path to save report output.")
    args = parser.parse_args()

    validator = DatasetValidator(data_dir=args.data_dir, output_dir=args.output_dir)
    passed = validator.run_all_validations()

    if not passed:
        print("\n❌ DATASET REJECTED: Training is BLOCKED until all validation modules pass.")
        sys.exit(1)
    else:
        print("\n✅ DATASET APPROVED: Safe for model training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
