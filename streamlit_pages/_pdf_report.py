"""
_pdf_report.py  — Premium Clinical PDF Report Generator
"""
from io import BytesIO
from datetime import datetime

def _try_import():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        return True
    except ImportError:
        return False

AVAILABLE = _try_import()


def _build_base_styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    styles = getSampleStyleSheet()
    title  = ParagraphStyle('MyTitle',  parent=styles['Title'],
                            fontSize=24, textColor=colors.HexColor('#4c1d95'),
                            spaceAfter=4, leading=30)
    sub    = ParagraphStyle('MySub',    parent=styles['Normal'],
                            fontSize=10, textColor=colors.HexColor('#7c3aed'),
                            spaceAfter=2)
    meta   = ParagraphStyle('MyMeta',  parent=styles['Normal'],
                            fontSize=9, textColor=colors.HexColor('#6b7280'),
                            spaceAfter=14)
    h2     = ParagraphStyle('MyH2',    parent=styles['Heading2'],
                            fontSize=12, textColor=colors.HexColor('#1e1b4b'),
                            spaceBefore=16, spaceAfter=6,
                            borderPadding=(4,0,4,8))
    body   = ParagraphStyle('MyBody',  parent=styles['Normal'],
                            fontSize=10, leading=16,
                            textColor=colors.HexColor('#111827'))
    warn   = ParagraphStyle('MyWarn',  parent=styles['Normal'],
                            fontSize=9, textColor=colors.HexColor('#92400e'),
                            backColor=colors.HexColor('#fef3c7'),
                            borderPadding=8, leading=14)
    bullet = ParagraphStyle('MyBullet',parent=styles['Normal'],
                            fontSize=9.5, leading=15,
                            textColor=colors.HexColor('#1f2937'),
                            leftIndent=12)
    return dict(title=title, sub=sub, meta=meta, h2=h2,
                body=body, warn=warn, bullet=bullet)


def _header_table(title_text, subtitle_text):
    """Returns a styled purple header table block."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle

    h_title = ParagraphStyle('HT', fontSize=16, fontName='Helvetica-Bold',
                             textColor=colors.white, leading=22)
    h_sub   = ParagraphStyle('HS', fontSize=9,
                             textColor=colors.HexColor('#e9d5ff'), leading=13)

    data = [
        [Paragraph(title_text,    h_title)],
        [Paragraph(subtitle_text, h_sub)],
    ]
    tbl = Table(data, colWidths=[17.0 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#4c1d95')),
        ('BACKGROUND',   (0, 1), (-1, -1), colors.HexColor('#6d28d9')),
        ('TOPPADDING',   (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 10),
        ('LEFTPADDING',  (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
    ]))
    return tbl


def generate_clinical_pdf(patient: dict, result: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.6*cm, bottomMargin=1.8*cm)
    S = _build_base_styles()

    RESULT_CONFIG = {
        'CN':   ('#166534', '#dcfce7', '#22c55e', 'Cognitively Normal',
                 'No significant cognitive impairment detected. Brain function appears '
                 'within normal limits. Continue routine annual cognitive screenings '
                 'and maintain a brain-healthy lifestyle.'),
        'LMCI': ('#92400e', '#fef3c7', '#f59e0b', 'Mild Cognitive Impairment',
                 'Mild Cognitive Impairment (MCI) detected — a transitional stage between '
                 'normal aging and early dementia. Close monitoring every 6 months is '
                 'recommended. Cognitive rehabilitation and lifestyle interventions may help.'),
        'AD':   ('#991b1b', '#fee2e2', '#ef4444', "Alzheimer's Disease",
                 'Findings consistent with Alzheimer\'s Disease. Immediate specialist referral '
                 'is strongly advised. A comprehensive care plan including medication review, '
                 'caregiver support, and safety assessments should be initiated promptly.'),
    }
    RECOMMENDATIONS = {
        'CN':   ['-  Annual cognitive screening',
                 '-  Mediterranean / MIND diet',
                 '-  150 min/week moderate aerobic exercise',
                 '-  Mental stimulation: puzzles, reading, social activities',
                 '-  Manage cardiovascular risk factors (BP, cholesterol)'],
        'LMCI': ['!  Follow-up neuropsychological evaluation in 6 months',
                 '!  Referral to a memory clinic',
                 '!  Brain MRI scan recommended',
                 '!  Monitor medications that may worsen cognition',
                 '!  Discuss cognitive training programs',
                 '!  Caregiver education and support resources'],
        'AD':   ['*  Urgent neurology referral',
                 '*  Full neuropsychological battery assessment',
                 '*  MRI / PET scan for structural evaluation',
                 '*  Review and adjust current medications',
                 '*  Initiate cholinesterase inhibitor therapy (if appropriate)',
                 '*  Comprehensive care and safety plan',
                 '*  Legal and financial planning guidance for family'],
    }

    txt_col, bg_col, accent, label, desc = RESULT_CONFIG[result]
    recs = RECOMMENDATIONS[result]
    now  = datetime.now().strftime("%d %B %Y, %I:%M %p")
    mmse = int(patient.get('mmse') or 0)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(_header_table(
        "Alzheimer's Prediction System",
        "AI-Powered Clinical Decision Support  -  Final Year Research Project"
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("CLINICAL ASSESSMENT REPORT", S['sub']))
    story.append(Paragraph(f"Report generated: {now}", S['meta']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#4c1d95'), spaceAfter=6))

    # ── Patient Info table ────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", S['h2']))
    val_style = __import__('reportlab.lib.styles', fromlist=['ParagraphStyle']).ParagraphStyle('val_style', parent=S['body'], fontSize=9, leading=11, textColor=colors.HexColor('#4c1d95'))
    def pval(text): return Paragraph(str(text), val_style)
    info_data = [
        ["Field", "Value", "Field", "Value"],
        ["Age",              pval(f"{patient.get('age','N/A')} yrs"),
         "Gender",           pval(patient.get('gender','N/A'))],
        ["Education",        pval(f"{patient.get('education','N/A')} yrs"),
         "MMSE Score",       pval(f"{mmse} / 30")],
        ["Ethnicity",        pval(patient.get('ethnicity','N/A')),
         "Race",             pval(patient.get('race','N/A'))],
        ["APOE4 Alleles",    pval(patient.get('apoe4','N/A')),
         "APOE Genotype",    pval(patient.get('apoe_genotype','N/A'))],
        ["Imputed Genotype", pval(patient.get('imputed_genotype','N/A')),
         "Report Date",      pval(datetime.now().strftime("%d %b %Y"))],
    ]
    tbl = Table(info_data, colWidths=[3.8*cm, 4.6*cm, 3.8*cm, 4.6*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
         [colors.HexColor('#f5f3ff'), colors.HexColor('#ede9fe')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#d1d5db')),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (2,1), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1), (0,-1), colors.HexColor('#4c1d95')),
        ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor('#4c1d95')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

    # ── Result Banner (Table-based, no unsupported style params) ─────────────
    story.append(Paragraph("Prediction Result", S['h2']))
    res_data = [[f"{result}  —  {label}"]]
    rtbl = Table(res_data, colWidths=[17.2*cm])
    rtbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor(bg_col)),
        ('TEXTCOLOR',     (0,0), (-1,-1), colors.HexColor(txt_col)),
        ('FONTNAME',      (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 14),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('BOX',           (0,0), (-1,-1), 2, colors.HexColor(accent)),
    ]))
    story.append(rtbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(desc, S['body']))

    # ── MMSE ──────────────────────────────────────────────────────────────────
    story.append(Paragraph("MMSE Score Interpretation", S['h2']))
    if mmse >= 24:   mi = f"Score {mmse}/30  —  Normal cognitive function"
    elif mmse >= 19: mi = f"Score {mmse}/30  —  Mild cognitive impairment range"
    elif mmse >= 10: mi = f"Score {mmse}/30  —  Moderate dementia range"
    else:            mi = f"Score {mmse}/30  —  Severe dementia range"
    story.append(Paragraph(mi, S['body']))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Recommendations", S['h2']))
    for rec in recs:
        story.append(Paragraph(rec, S['bullet']))
        story.append(Spacer(1, 2))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor('#d1d5db'), spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for research and educational "
        "purposes only. It does NOT constitute a clinical medical diagnosis. "
        "Always consult a qualified neurologist or healthcare professional for clinical decisions.",
        S['warn']))

    # ── Footer row ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    footer_data = [["Alzheimer's Prediction System",
                    "AI · Clinical · MRI Deep Learning",
                    f"Generated: {datetime.now().strftime('%d %b %Y')}"]]
    ftbl = Table(footer_data, colWidths=[6*cm, 6*cm, 5.6*cm])
    ftbl.setStyle(TableStyle([
        ('FONTSIZE',     (0,0), (-1,-1), 7.5),
        ('TEXTCOLOR',    (0,0), (-1,-1), colors.HexColor('#9ca3af')),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('ALIGN',        (2,0), (2,0), 'RIGHT'),
    ]))
    story.append(ftbl)

    doc.build(story)
    return buf.getvalue()


def generate_mri_pdf(class_name: str, confidence: float,
                     probs: list, inv_idx: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.6*cm, bottomMargin=1.8*cm)
    S = _build_base_styles()
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    STAGE = {
        "Non Demented":       ('#166534','#dcfce7','#22c55e',
                               'No signs of dementia detected. Brain structures appear within '
                               'normal limits. Continue routine annual check-ups.'),
        "Very Mild Demented": ('#3f6212','#ecfccb','#84cc16',
                               'Very early changes consistent with mild cognitive decline detected. '
                               'Recommend follow-up MRI in 6-12 months and neuropsychological evaluation.'),
        "Mild Demented":      ('#92400e','#fef3c7','#f59e0b',
                               'MRI findings consistent with mild Alzheimer\'s dementia. Referral to a '
                               'neurologist is recommended. Cognitive and functional assessments warranted.'),
        "Moderate Demented":  ('#991b1b','#fee2e2','#ef4444',
                               'MRI findings consistent with moderate-stage Alzheimer\'s disease. '
                               'Urgent specialist referral required. Comprehensive care planning advised.'),
    }

    txt_col, bg_col, accent, desc = STAGE.get(
        class_name, ('#1e1b4b','#f5f3ff','#7c3aed','Analysis complete.'))

    story = []

    # Header
    story.append(_header_table(
        "Alzheimer's Prediction System",
        "AI-Powered MRI Brain Scan Analysis  -  CNN Deep Learning Report"
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("MRI CNN ANALYSIS REPORT", S['sub']))
    story.append(Paragraph(f"Report generated: {now}", S['meta']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#4c1d95'), spaceAfter=6))

    # ── Patient / Scan Info table ─────────────────────────────────────────────
    import random
    scan_id = f"MRI-{random.randint(100000, 999999)}"
    story.append(Paragraph("Scan Information", S['h2']))
    val_style = __import__('reportlab.lib.styles', fromlist=['ParagraphStyle']).ParagraphStyle('val_style', parent=S['body'], fontSize=9, leading=11, textColor=colors.HexColor('#4c1d95'))
    def pval(text): return Paragraph(str(text), val_style)
    info_data = [
        ["Field", "Value", "Field", "Value"],
        ["Scan ID",          pval(scan_id),
         "Scan Modality",    pval("T1-Weighted MRI")],
        ["Patient Name",     pval("[ Not Provided in Demo ]"),
         "Resolution",       pval("High (128x128 3-channel)")],
        ["Referring Dept.",  pval("Neurology & Imaging"),
         "Report Date",      pval(datetime.now().strftime("%d %b %Y"))],
    ]
    stbl = Table(info_data, colWidths=[3.8*cm, 4.6*cm, 3.8*cm, 4.6*cm])
    stbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
         [colors.HexColor('#f5f3ff'), colors.HexColor('#ede9fe')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#d1d5db')),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (2,1), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1), (0,-1), colors.HexColor('#4c1d95')),
        ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor('#4c1d95')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(stbl)
    story.append(Spacer(1, 6))

    # Result banner
    story.append(Paragraph("CNN Prediction Result", S['h2']))
    res_data = [[f"{class_name}   -   Confidence: {confidence:.1f}%"]]
    rtbl = Table(res_data, colWidths=[17.2*cm])
    rtbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor(bg_col)),
        ('TEXTCOLOR',     (0,0), (-1,-1), colors.HexColor(txt_col)),
        ('FONTNAME',      (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 15),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('BOX',           (0,0), (-1,-1), 1.5, colors.HexColor(accent)),
    ]))
    story.append(rtbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph(desc, S['body']))

    # Probability breakdown table
    from reportlab.graphics.shapes import Drawing, Rect
    story.append(Paragraph("Class Probability Breakdown", S['h2']))
    prob_data = [["Alzheimer's Stage", "Probability", "Confidence Bar"]]
    for idx_num in sorted(inv_idx.keys()):
        name = inv_idx[idx_num]
        pct  = float(probs[idx_num]) * 100
        
        # Create a graphical progress bar instead of text to prevent any text merging issues
        d = Drawing(140, 12)
        # Background bar
        d.add(Rect(0, 1, 140, 10, fillColor=colors.HexColor('#e5e7eb'), strokeColor=None, rx=3, ry=3))
        # Fill bar
        bar_width = max(2, (pct / 100.0) * 140)
        d.add(Rect(0, 1, bar_width, 10, fillColor=colors.HexColor('#6d28d9'), strokeColor=None, rx=3, ry=3))
        
        prob_data.append([name, f"{pct:.1f}%", d])

    ptbl = Table(prob_data, colWidths=[6.5*cm, 3.5*cm, 7*cm])
    ptbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
         [colors.HexColor('#f5f3ff'), colors.HexColor('#ede9fe')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#d1d5db')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME',      (1,1), (1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1,1), (1,-1), colors.HexColor('#4c1d95')),
    ]))
    story.append(ptbl)
    story.append(Spacer(1, 14))

    # ── Anatomical Context ───────────────────────────────────────────────────
    story.append(Paragraph("Anatomical Context & Biomarkers", S['h2']))
    ctx_text = (
        "This AI evaluation involves analyzing structural biomarkers across the brain volume via a Convolutional Neural Network. "
        "Key areas evaluated include the <b>Hippocampus</b> (critical for memory formation), the <b>Cerebral Cortex</b> "
        "(responsible for cognitive processing), and <b>Ventricle structures</b> (which expand as surrounding tissue shrinks). "
        f"The prediction of <i>{class_name}</i> indicates that the volumetric and structural patterns identified "
        "strongly correlate with this stage of neurodegeneration."
    )
    story.append(Paragraph(ctx_text, S['body']))
    story.append(Spacer(1, 8))

    # ── Personalized Care & Recommendations ──────────────────────────────────
    MRI_RECOMMENDATIONS = {
        "Non Demented": {
            "doctor": ["Routine annual cognitive screenings are recommended.",
                       "Monitor for any subjective changes in memory or cognition."],
            "diet": ["Maintain a brain-healthy diet such as the Mediterranean or MIND diet.",
                     "Rich in Omega-3 fatty acids, antioxidants, and whole grains.",
                     "Stay hydrated and limit processed foods and refined sugars."],
            "lifestyle": ["Engage in regular aerobic exercise (e.g., 150 mins/week).",
                          "Keep the brain active with puzzles, reading, or learning new skills.",
                          "Ensure 7-8 hours of quality sleep per night."]
        },
        "Very Mild Demented": {
            "doctor": ["Schedule a follow-up MRI in 6-12 months.",
                       "Referral for a comprehensive neuropsychological evaluation.",
                       "Review current medications that might affect cognition."],
            "diet": ["Strictly follow the MIND diet to support cognitive function.",
                     "Increase intake of leafy greens, berries, and nuts.",
                     "Consider consulting a nutritionist for a personalized plan."],
            "lifestyle": ["Introduce memory aids like calendars, reminders, and notes.",
                          "Maintain social engagement and group activities.",
                          "Establish a structured daily routine to reduce cognitive load."]
        },
        "Mild Demented": {
            "doctor": ["Neurologist referral is highly recommended for clinical confirmation.",
                       "Consider initiation of approved cognitive-enhancing medications.",
                       "Discuss caregiver support and future care planning."],
            "diet": ["Ensure adequate nutrition and hydration; monitor weight changes.",
                     "Prepare easy-to-eat, nutritious meals.",
                     "Omega-3 supplements may be discussed with a physician."],
            "lifestyle": ["Simplify the living environment to ensure safety and reduce fall risks.",
                          "Encourage light physical activities like walking under supervision.",
                          "Join a support group for both the patient and caregivers."]
        },
        "Moderate Demented": {
            "doctor": ["Urgent neurologist consultation for advanced disease management.",
                       "Comprehensive safety assessment of the living environment.",
                       "Evaluate for behavioral or psychological symptoms of dementia (BPSD)."],
            "diet": ["Supervise meals to ensure proper chewing and swallowing.",
                     "Offer finger foods if utensils become difficult to use.",
                     "Maintain hydration with frequent offerings of water or juices."],
            "lifestyle": ["Implement 24/7 supervision or consider a memory care facility.",
                          "Use identification bracelets and door alarms to prevent wandering.",
                          "Focus on providing a calm, reassuring, and familiar environment."]
        }
    }

    story.append(Paragraph("Care Plan & Clinical Suggestions", S['h2']))
    recs = MRI_RECOMMENDATIONS.get(class_name, MRI_RECOMMENDATIONS["Non Demented"])
    
    # Custom styles for the recommendations
    res_style_p = __import__('reportlab.lib.styles', fromlist=['ParagraphStyle']).ParagraphStyle
    heading_style = res_style_p('RecH', parent=S['body'], fontName='Helvetica-Bold', textColor=colors.HexColor('#4c1d95'), spaceBefore=8, spaceAfter=4, fontSize=11)
    item_style = res_style_p('RecI', parent=S['bullet'], fontSize=10, leading=16, spaceAfter=2)

    # Doctor Tips
    story.append(Paragraph("Medical & Clinical Next Steps", heading_style))
    for tip in recs["doctor"]:
        story.append(Paragraph(f"-  {tip}", item_style))
    
    # Diet
    story.append(Paragraph("Dietary Suggestions", heading_style))
    for tip in recs["diet"]:
        story.append(Paragraph(f"-  {tip}", item_style))
    
    # Lifestyle
    story.append(Paragraph("Lifestyle & Wellbeing", heading_style))
    for tip in recs["lifestyle"]:
        story.append(Paragraph(f"-  {tip}", item_style))
    story.append(Spacer(1, 12))

    # ── Glossary ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Terminology", S['h2']))
    glossary = [
        "<b>Mild Cognitive Impairment (MCI):</b> A transitional stage between expected cognitive decline of normal aging and dementia.",
        "<b>Atrophy:</b> The shrinking of brain tissue due to the loss of neurons.",
        "<b>Ventricles:</b> Fluid-filled spaces in the brain that often appear larger on an MRI when surrounding brain tissue shrinks."
    ]
    for term in glossary:
        story.append(Paragraph(f"-  {term}", item_style))
    story.append(Spacer(1, 12))

    # ── Neurologist Notes ────────────────────────────────────────────────────
    story.append(Paragraph("Attending Physician / Neurologist Notes", S['h2']))
    for _ in range(3):
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#9ca3af'), spaceAfter=0))
    story.append(Spacer(1, 14))

    # Model info
    story.append(Paragraph("Technical Model Information", S['h2']))
    model_info = [
        ["Parameter", "Value"],
        ["Model",          "MobileNetV3 / EfficientNet-B0 (PyTorch)"],
        ["Dataset",        "OASIS - 39,986 MRI Brain Scans"],
        ["Classes",        "4 (Non / Very Mild / Mild / Moderate Demented)"],
        ["Validation Acc", "89.6%"],
        ["Input Size",     "128 x 128 x 3 RGB"],
        ["Framework",      "PyTorch"],
    ]
    mtbl = Table(model_info, colWidths=[5*cm, 12.6*cm])
    mtbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#f5f3ff'), colors.HexColor('#ede9fe')]),
        ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#d1d5db')),
        ('FONTNAME',     (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',    (0,1), (0,-1), colors.HexColor('#4c1d95')),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
    ]))
    story.append(mtbl)

    # Disclaimer
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor('#d1d5db'), spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This MRI analysis is performed by a CNN model trained on the OASIS dataset "
        "for research purposes only. It does NOT replace a clinical radiological report. "
        "Consult a qualified radiologist and neurologist for clinical decisions.",
        S['warn']))

    story.append(Spacer(1, 8))
    footer_data = [["Alzheimer's Prediction System",
                    "CNN - PyTorch - OASIS Dataset",
                    f"Generated: {datetime.now().strftime('%d %b %Y')}"]]
    ftbl = Table(footer_data, colWidths=[6*cm, 6*cm, 5.6*cm])
    ftbl.setStyle(TableStyle([
        ('FONTSIZE',  (0,0), (-1,-1), 7.5),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#9ca3af')),
        ('TOPPADDING',(0,0), (-1,-1), 4),
        ('ALIGN',     (2,0), (2,0), 'RIGHT'),
    ]))
    story.append(ftbl)

    doc.build(story)
    return buf.getvalue()
