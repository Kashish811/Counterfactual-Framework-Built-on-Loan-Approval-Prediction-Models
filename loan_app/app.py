# app.py — LoanSense Backend v2
# Features: SHAP, Feature Importance, Borderline Alert,
#           Progress Tracker, Reapplication Score, What-If, PDF Report
# Run: python app.py  →  http://localhost:5000

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import io
import traceback
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

app = Flask(__name__)
CORS(app)

# ── Load model + threshold ────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
model     = pickle.load(open(os.path.join(BASE_DIR, "loan_model.pkl"),     "rb"))
threshold = pickle.load(open(os.path.join(BASE_DIR, "loan_threshold.pkl"), "rb"))
threshold = max(threshold, 0.45)
print(f"[✓] Model loaded  |  Threshold = {threshold:.4f}")

# ── Feature definitions ───────────────────────────────────────
FEATURE_ORDER = [
    "checking_account_status", "duration_months", "credit_history",
    "purpose", "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker",
]

NUMERICAL_FEATURES = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]

SAFE_FEATURES_TO_VARY = [
    "credit_amount", "duration_months", "installment_rate",
    "purpose", "existing_credits", "other_debtors",
    "other_installment_plans", "property",
    "employment_since", "residence_since", "housing",
]

READABLE = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0–200 DM",
        "A13": "≥ 200 DM", "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits / all paid", "A31": "All paid duly",
        "A32": "Existing paid duly", "A33": "Delay in payments",
        "A34": "Critical account",
    },
    "purpose": {
        "A40": "New car", "A41": "Used car", "A42": "Furniture",
        "A43": "Radio / TV", "A44": "Appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business",
        "A410": "Other",
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM",
        "A64": "≥ 1000 DM", "A65": "Unknown / none",
    },
    "employment_since": {
        "A71": "Unemployed", "A72": "< 1 year", "A73": "1–4 years",
        "A74": "4–7 years", "A75": "≥ 7 years",
    },
    "other_debtors":           {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "property": {
        "A121": "Real estate", "A122": "Life insurance",
        "A123": "Car or other", "A124": "No property",
    },
    "housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "personal_status_sex": {
        "A91": "Male divorced/sep", "A92": "Female divorced/married",
        "A93": "Male single", "A94": "Male married/widowed",
    },
    "job": {
        "A171": "Unskilled non-resident", "A172": "Unskilled resident",
        "A173": "Skilled employee", "A174": "Management/highly qualified",
    },
    "telephone":     {"A191": "None", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}

INSIGHTS = {
    "credit_amount": {
        "why": "A lower loan amount reduces your debt burden ratio, making you significantly less risky to the lender.",
        "how": "Consider reducing the loan scope, splitting into smaller loans, or saving up a portion before applying.",
        "timeline": "Immediate — adjust your application amount", "priority": 1,
    },
    "duration_months": {
        "why": "Shorter loan terms signal lower long-term risk to the lender and reduce total interest exposure.",
        "how": "Opt for a shorter repayment period if your monthly income can support higher installments.",
        "timeline": "Immediate — adjust your application term", "priority": 2,
    },
    "installment_rate": {
        "why": "A lower installment rate means smaller monthly payments relative to your income.",
        "how": "Choose a plan with lower monthly installments, or increase your declared income source.",
        "timeline": "Immediate — select a different repayment plan", "priority": 2,
    },
    "purpose": {
        "why": "Lenders view certain loan purposes as lower risk. Productive purposes are preferred over 'other'.",
        "how": "If possible, reframe your loan purpose to match a lower-risk category that fits your actual need.",
        "timeline": "Immediate — recategorize your application", "priority": 3,
    },
    "other_installment_plans": {
        "why": "Existing installment plans increase your total debt obligations, reducing approval chances.",
        "how": "Pay off or close existing installment plans before reapplying.",
        "timeline": "Short-term — 1 to 3 months", "priority": 1,
    },
    "other_debtors": {
        "why": "Having a guarantor or co-applicant reduces the lender's risk by providing a financial backup.",
        "how": "Ask a financially stable family member or friend to co-sign your loan application.",
        "timeline": "Immediate — add to your application", "priority": 2,
    },
    "property": {
        "why": "Owning property acts as collateral, significantly increasing lender confidence.",
        "how": "If you own any property (real estate, vehicle, life insurance), declare it in your application.",
        "timeline": "Immediate — update your application", "priority": 2,
    },
    "housing": {
        "why": "Housing status affects perceived financial stability. Owned housing is viewed most favourably.",
        "how": "Ensure your housing status is accurately declared in your application.",
        "timeline": "Immediate — verify your application details", "priority": 3,
    },
    "employment_since": {
        "why": "Longer employment history signals income stability — one of the strongest approval signals.",
        "how": "If you recently changed jobs, wait 6–12 months at your current role before reapplying.",
        "timeline": "Medium-term — 6 to 12 months", "priority": 1,
    },
    "residence_since": {
        "why": "Longer residence at the same address signals personal stability to the lender.",
        "how": "Avoid moving before reapplying. 2+ years at the same address is viewed positively.",
        "timeline": "Medium-term — 1 to 2 years", "priority": 3,
    },
    "existing_credits": {
        "why": "Fewer existing credits at the bank indicates you are not over-leveraged.",
        "how": "Pay off and close any existing credit accounts before reapplying.",
        "timeline": "Short-term — 1 to 3 months", "priority": 2,
    },
}

CF_RANKS = {
    "employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "property":         ["A124", "A123", "A122", "A121"],
}

FALLBACKS = {"personal_status_sex": {"A95": "A92"}}

# ── Helpers ───────────────────────────────────────────────────
def to_df(data):
    row = {f: data.get(f) for f in FEATURE_ORDER}
    for feat, fb in FALLBACKS.items():
        if row.get(feat) in fb:
            row[feat] = fb[row[feat]]
    df = pd.DataFrame([row])
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def lbl(feat, val):
    return READABLE.get(feat, {}).get(str(val), str(val))

def is_valid_cf(orig, cf_row):
    for feat, ranking in CF_RANKS.items():
        ov, cv = str(orig[feat].iloc[0]), str(cf_row[feat])
        if ov in ranking and cv in ranking:
            if ranking.index(cv) < ranking.index(ov):
                return False
    return True

def has_changes(orig, cf_row):
    return any(str(orig[f].iloc[0]) != str(cf_row[f]) for f in SAFE_FEATURES_TO_VARY)

def build_plan(orig, cf_row, plan_num):
    steps = []
    for feat in SAFE_FEATURES_TO_VARY:
        ov, cv = orig[feat].iloc[0], cf_row[feat]
        if str(ov) != str(cv):
            ins = INSIGHTS.get(feat, {})
            steps.append({
                "feature":  feat.replace("_", " ").title(),
                "feat_key": feat,
                "from":     lbl(feat, ov),
                "to":       lbl(feat, cv),
                "to_raw":   cv,
                "why":      ins.get("why",      "This change improves your approval profile."),
                "how":      ins.get("how",      "Update your application accordingly."),
                "timeline": ins.get("timeline", "As soon as possible"),
                "priority": ins.get("priority", 3),
            })
            
    import random
    pad_mutations = {
        "duration_months": lambda x: max(4, int(float(x) * 0.85)),
        "credit_amount": lambda x: max(250, int(float(x) * 0.9)),
        "employment_since": lambda x: "A75",
        "property": lambda x: "A121",
        "housing": lambda x: "A152",
        "residence_since": lambda x: min(4, int(x) + 1),
        "existing_credits": lambda x: max(1, int(x) - 1),
        "purpose": lambda x: "A40",
        "installment_rate": lambda x: max(1, int(x) - 1),
        "other_debtors": lambda x: "A103",
        "other_installment_plans": lambda x: "A143"
    }
    
    unchanged = [f for f in SAFE_FEATURES_TO_VARY if str(orig[f].iloc[0]) == str(cf_row[f])]
    random.shuffle(unchanged)
    
    for feat in unchanged:
        if len(steps) >= 5: break
        if feat in pad_mutations:
            ov = orig[feat].iloc[0]
            nv = pad_mutations[feat](ov)
            if str(ov) != str(nv) and is_valid_cf(orig, {feat: nv, **{f: cf_row[f] for f in cf_row.keys()}}):
                ins = INSIGHTS.get(feat, {})
                steps.append({
                    "feature":  feat.replace("_", " ").title(),
                    "feat_key": feat,
                    "from":     lbl(feat, ov),
                    "to":       lbl(feat, nv),
                    "to_raw":   nv,
                    "why":      ins.get("why", "This additional change further secures your approval."),
                    "how":      ins.get("how", "Update your application accordingly."),
                    "timeline": ins.get("timeline", "As soon as possible"),
                    "priority": 4
                })

    steps.sort(key=lambda x: x["priority"])
    return {"plan": plan_num, "steps": steps}

# ── Feature importance chart ──────────────────────────────────
def get_feature_importance_chart():
    try:
        rf   = model.named_steps["classifier"]
        pre  = model.named_steps["preprocessing"]
        cat  = pre.named_transformers_["cat"]
        num_names = NUMERICAL_FEATURES
        cat_names = cat.get_feature_names_out(
            [f for f in FEATURE_ORDER if f not in NUMERICAL_FEATURES]
        ).tolist()
        all_names = num_names + cat_names
        importances = rf.feature_importances_

        # Group by original feature
        feat_imp = {}
        for name, imp in zip(all_names, importances):
            # cat names look like "checking_account_status_A12"
            orig = name.split("_A")[0] if "_A" in name else name
            # handle numeric feature names directly
            if orig in NUMERICAL_FEATURES:
                orig = name
            feat_imp[orig] = feat_imp.get(orig, 0) + float(imp)

        # Top 10
        sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        names  = [x[0].replace("_", " ").title() for x in sorted_fi]
        values = [x[1] for x in sorted_fi]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#c8a96e" if v == max(values) else "#3a3a45" for v in values]
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="none")
        ax.set_xlabel("Importance Score", color="#9998a8", fontsize=11)
        ax.set_title("Top 10 Feature Importances", color="#f0eff0", fontsize=13, pad=12)
        ax.tick_params(colors="#9998a8")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#141417")
        fig.patch.set_facecolor("#141417")
        ax.xaxis.label.set_color("#9998a8")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#141417")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[feature importance error] {e}")
        return None

# ── SHAP chart ────────────────────────────────────────────────
def get_shap_chart(df_row):
    try:
        import shap
        rf  = model.named_steps["classifier"]
        pre = model.named_steps["preprocessing"]
        cat = pre.named_transformers_["cat"]

        X_transformed = pre.transform(df_row)
        num_names = NUMERICAL_FEATURES
        cat_names = cat.get_feature_names_out(
            [f for f in FEATURE_ORDER if f not in NUMERICAL_FEATURES]
        ).tolist()
        all_names = num_names + cat_names

        explainer   = shap.TreeExplainer(rf)
        shap_vals   = explainer.shap_values(X_transformed)

        # class 1 = approved
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        # Group by original feature
        feat_shap = {}
        for name, val in zip(all_names, sv):
            orig = name.split("_A")[0] if "_A" in name else name
            if orig in NUMERICAL_FEATURES:
                orig = name
            scalar_val = float(np.array(val).flatten()[0])
            feat_shap[orig] = feat_shap.get(orig, 0) + scalar_val

        sorted_shap = sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        names  = [x[0].replace("_", " ").title() for x in sorted_shap]
        values = [float(x[1]) for x in sorted_shap]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#3ecf8e" if v > 0 else "#f87171" for v in values]
        ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="none")
        ax.axvline(0, color="#6e6d7a", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on approval probability)", color="#9998a8", fontsize=10)
        ax.set_title("Feature Impact on This Decision (SHAP)", color="#f0eff0", fontsize=13, pad=12)
        ax.tick_params(colors="#9998a8")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#141417")
        fig.patch.set_facecolor("#141417")

        green_patch = mpatches.Patch(color="#3ecf8e", label="Helped approval")
        red_patch   = mpatches.Patch(color="#f87171", label="Hurt approval")
        ax.legend(handles=[green_patch, red_patch], facecolor="#1c1c21",
                  labelcolor="#9998a8", fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#141417")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[SHAP error] {e}")
        traceback.print_exc()
        return None

# ── Reapplication score ───────────────────────────────────────
def get_reapplication_score(orig_prob, action_plans, df_row, data):
    if not action_plans:
        return {"current": round(orig_prob*100, 1), "projected": None, "improvement": None, "steps": 0}
    try:
        best_plan = action_plans[0]
        trial = data.copy()
        for step in best_plan["steps"]:
            feat_key = step.get("feat_key")
            if feat_key and feat_key in SAFE_FEATURES_TO_VARY:
                trial[feat_key] = step["to_raw"] if "to_raw" in step else step["to"]
        trial_df  = to_df(trial)
        new_prob  = float(model.predict_proba(trial_df)[:, 1][0])
        return {
            "current":     round(orig_prob * 100, 1),
            "projected":   round(new_prob  * 100, 1),
            "improvement": round((new_prob - orig_prob) * 100, 1),
            "steps":       len(best_plan["steps"]),
        }
    except Exception as e:
        print(f"[reapplication score error] {e}")
        return {"current": round(orig_prob*100, 1), "projected": None, "improvement": None, "steps": 0}

# ── PDF report ────────────────────────────────────────────────
def generate_pdf(data, verdict, probability, action_plans, shap_b64, borderline):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
        import datetime

        buf = io.BytesIO()
        c   = canvas.Canvas(buf, pagesize=A4)
        W, H = A4

        # Header bar
        c.setFillColor(colors.HexColor("#141417"))
        c.rect(0, H-80, W, 80, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#c8a96e"))
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, H-45, "LoanSense")
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 11)
        c.drawString(40, H-65, "Credit Assessment Report")
        c.setFont("Helvetica", 9)
        c.drawRightString(W-40, H-50, datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"))

        y = H - 110

        # Verdict
        vcolor = colors.HexColor("#3ecf8e") if verdict else colors.HexColor("#f87171")
        c.setFillColor(vcolor)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, y, "LOAN APPROVED" if verdict else "LOAN REJECTED")

        c.setFillColor(colors.HexColor("#6e6d7a"))
        c.setFont("Helvetica", 11)
        c.drawString(40, y-20, f"Approval probability: {probability}%")

        if borderline:
            c.setFillColor(colors.HexColor("#f59e0b"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(40, y-38, "⚠ Borderline case — small changes could flip this decision")
            y -= 55
        else:
            y -= 45

        # Divider
        c.setStrokeColor(colors.HexColor("#2a2a30"))
        c.line(40, y, W-40, y)
        y -= 20

        # SHAP chart
        if shap_b64:
            c.setFillColor(colors.HexColor("#141417"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Feature Impact on This Decision (SHAP)")
            y -= 8
            img_data = base64.b64decode(shap_b64)
            img      = ImageReader(io.BytesIO(img_data))
            img_h    = 160
            c.drawImage(img, 40, y-img_h, width=W-80, height=img_h)
            y -= img_h + 20

        # Action plans
        c.line(40, y, W-40, y)
        y -= 20
        c.setFillColor(colors.HexColor("#141417"))
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "Your Personalised Action Plan")
        y -= 25

        if action_plans:
            for i, plan in enumerate(action_plans[:2]):
                c.setFillColor(colors.HexColor("#c8a96e"))
                c.setFont("Helvetica-Bold", 11)
                c.drawString(40, y, f"Plan {i+1}")
                y -= 18

                for step in plan["steps"]:
                    if y < 80:
                        c.showPage()
                        y = H - 60
                    c.setFillColor(colors.HexColor("#1c1c21"))
                    c.roundRect(40, y-45, W-80, 50, 4, fill=1, stroke=0)

                    c.setFillColor(colors.HexColor("#f0eff0"))
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(50, y-12, step["feature"])

                    c.setFillColor(colors.HexColor("#f87171"))
                    c.setFont("Helvetica", 9)
                    c.drawString(50, y-25, f"{step['from']}")
                    c.setFillColor(colors.HexColor("#6e6d7a"))
                    c.drawString(50 + c.stringWidth(step['from'], "Helvetica", 9) + 5, y-25, "→")
                    c.setFillColor(colors.HexColor("#3ecf8e"))
                    c.drawString(50 + c.stringWidth(step['from'], "Helvetica", 9) + 18, y-25, f"{step['to']}")

                    c.setFillColor(colors.HexColor("#9998a8"))
                    c.setFont("Helvetica", 8)
                    c.drawString(50, y-37, f"Timeline: {step['timeline']}")
                    y -= 58

                y -= 10
        else:
            c.setFillColor(colors.HexColor("#6e6d7a"))
            c.setFont("Helvetica", 11)
            c.drawString(40, y, "No specific plans could be generated. Please consult a financial advisor.")
            y -= 30

        # Footer
        c.setFillColor(colors.HexColor("#6e6d7a"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(W/2, 30, "Generated by LoanSense · For informational purposes only · Not financial advice")

        c.save()
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"[PDF error] {e}")
        traceback.print_exc()
        return None

# ── DiCE runner ───────────────────────────────────────────────
def run_dice(df_row, data, total_cfs):
    import dice_ml
    from dice_ml import Dice
    
    csv_path = os.path.join(BASE_DIR, "german_credit.csv")
    X_ref = pd.read_csv(csv_path)
    df_ref = X_ref.copy()
    df_ref["approved"] = model.predict(X_ref)

    dice_data = dice_ml.Data(
        dataframe=df_ref,
        continuous_features=NUMERICAL_FEATURES,
        outcome_name="approved"
    )
    dice_mdl  = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(dice_data, dice_mdl, method="random")
    permitted_range = {
        "credit_amount":    [int(float(data["credit_amount"]) * 0.3),
                              int(float(data["credit_amount"]))],
        "duration_months":  [min(4, int(data["duration_months"])), max(6, int(data["duration_months"]))],
        "installment_rate": [1, 4],
    }
    return exp.generate_counterfactuals(
        df_row, total_CFs=total_cfs, desired_class=1,
        features_to_vary=SAFE_FEATURES_TO_VARY,
        permitted_range=permitted_range
    )

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/feature-importance")
def feature_importance():
    chart = get_feature_importance_chart()
    return jsonify({"chart": chart})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        df_row = to_df(data)

        prob     = float(model.predict_proba(df_row)[:, 1][0])
        approved = prob >= threshold

        # Borderline alert: 45–60% = borderline zone
        borderline = (0.40 <= prob <= 0.65)
        borderline_msg = None
        if borderline and not approved:
            borderline_msg = f"You are very close to approval ({round(prob*100,1)}%). Just 1–2 small changes could flip your outcome."
        elif borderline and approved:
            borderline_msg = f"Your approval is marginal ({round(prob*100,1)}%). Strengthening your profile will make your application more secure."

        # SHAP
        shap_chart = get_shap_chart(df_row)

        if approved:
            reapp = {"current": round(prob*100,1), "projected": None, "improvement": None, "steps": 0}
            return jsonify({
                "approved":       True,
                "probability":    round(prob*100, 1),
                "borderline":     borderline,
                "borderline_msg": borderline_msg,
                "shap_chart":     shap_chart,
                "action_plans":   [],
                "reapplication":  reapp,
            })

        # DiCE
        action_plans = []
        try:
            cf    = run_dice(df_row, data, 6)
            cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
            for _, row in cf_df.iterrows():
                if not has_changes(df_row, row): continue
                if not is_valid_cf(df_row, row): continue
                plan = build_plan(df_row, row, len(action_plans)+1)
                if plan["steps"]: action_plans.append(plan)

            if not action_plans:
                cf    = run_dice(df_row, data, 10)
                cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
                for _, row in cf_df.iterrows():
                    if not has_changes(df_row, row): continue
                    if not is_valid_cf(df_row, row): continue
                    plan = build_plan(df_row, row, len(action_plans)+1)
                    if plan["steps"]: action_plans.append(plan)
        except Exception as e:
            print(f"[DiCE error] {e}")

        reapp = get_reapplication_score(prob, action_plans, df_row, data)

        return jsonify({
            "approved":       False,
            "probability":    round(prob*100, 1),
            "borderline":     borderline,
            "borderline_msg": borderline_msg,
            "shap_chart":     shap_chart,
            "action_plans":   action_plans[:3],
            "reapplication":  reapp,
        })

    except Exception as e:
        print(f"[predict error] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/whatif", methods=["POST"])
def whatif():
    """Real-time probability update for What-If simulator."""
    try:
        data   = request.get_json(force=True)
        df_row = to_df(data)
        prob   = float(model.predict_proba(df_row)[:, 1][0])
        return jsonify({
            "probability": round(prob*100, 1),
            "approved":    prob >= threshold,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pdf", methods=["POST"])
def pdf_report():
    """Generate and return PDF report."""
    try:
        payload      = request.get_json(force=True)
        data         = payload.get("inputs", {})
        verdict      = payload.get("approved", False)
        probability  = payload.get("probability", 0)
        action_plans = payload.get("action_plans", [])
        shap_b64     = payload.get("shap_chart", None)
        borderline   = payload.get("borderline", False)

        df_row    = to_df(data)
        shap_b64  = shap_b64 or get_shap_chart(df_row)
        pdf_buf   = generate_pdf(data, verdict, probability,
                                  action_plans, shap_b64, borderline)

        if pdf_buf:
            return send_file(
                pdf_buf, mimetype="application/pdf",
                as_attachment=True,
                download_name="loansense_report.pdf"
            )
        return jsonify({"error": "PDF generation failed"}), 500
    except Exception as e:
        print(f"[pdf route error] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "threshold": float(threshold)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)