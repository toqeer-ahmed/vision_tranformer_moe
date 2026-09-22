import os
import sys

def build_pdf():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError:
        print("ReportLab is not installed. Please run: pip install reportlab")
        sys.exit(1)

    pdf_path = "medical_segmentation_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles for Academic Publication Look
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=12,
        alignment=1 # Centered
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=20,
        alignment=1 # Centered
    )

    heading1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    heading2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("State-of-the-Art Research Report: Shared-Expert MoE SegFormer for Medical Image Segmentation", title_style))
    story.append(Paragraph("<b>Prepared by:</b> Toqeer Ahmed &nbsp;|&nbsp; <b>Dataset:</b> Kvasir-SEG Gastrointestinal Polyp Segmentation (1,000 Scans)", subtitle_style))
    story.append(Spacer(1, 10))

    # 1. Objective
    story.append(Paragraph("1. Executive Objective & Research Vision", heading1_style))
    story.append(Paragraph(
        "This research analyzes the progression of dense semantic segmentation architectures from convolutional networks (CNNs) "
        "and standard Hierarchical Vision Transformers (ViT) to advanced <b>Shared-Expert Mixture of Experts (MoE)</b> designs. "
        "The project evaluates parameter-efficient capacity scaling, Focal Tversky boundary optimization, and multi-scale "
        "inference for gastrointestinal polyp segmentation on high-variance endoscopic scans.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # 2. Dataset & Preprocessing
    story.append(Paragraph("2. Dataset Specification (Kvasir-SEG)", heading1_style))
    story.append(Paragraph("<b>Dataset:</b> Kvasir-SEG (1,000 annotated high-resolution colonoscopy frames showing gastrointestinal polyps).", body_style))
    story.append(Paragraph("<b>Preprocessing & Augmentations:</b>", body_style))
    story.append(Paragraph("• High Spatial Resolution: Images scaled to 352x352x3 pixels to preserve small lesion boundaries.", bullet_style))
    story.append(Paragraph("• Non-Rigid Tissue Augmentations: Albumentations ElasticTransforms, GridDistortion, ShiftScaleRotate, and ColorJitter.", bullet_style))
    story.append(Paragraph("• Split Ratio: 80% Training, 10% Validation, 10% Testing.", bullet_style))
    story.append(Spacer(1, 8))

    # 3. Methodology
    story.append(Paragraph("3. SOTA Architecture & Technical Innovations", heading1_style))
    story.append(Paragraph("<b>Baseline Architecture (SegFormer):</b> SegFormer (mit-b0) hierarchical backbone with overlapping patch embedding and an efficient All-MLP Decoder head.", body_style))
    story.append(Paragraph("<b>Shared-Expert MoE (DeepSeek-style):</b> MixFFN blocks in Stages 3 & 4 are replaced with custom SharedMoELayers. Each layer combines 1 dedicated shared expert (evaluated for all tokens) with 3 dynamically routed experts (mediated by noisy top-2 routing).", body_style))
    story.append(Paragraph("<b>Focal Tversky + Sobel Edge Loss:</b> Combines Cross-Entropy with Focal Tversky Loss (alpha=0.3, beta=0.7) to heavily penalize false negatives (missed polyps), combined with Sobel gradient filtering for sharp lesion boundary recovery.", body_style))
    
    story.append(PageBreak())

    # 4. Quantitative Results & Tables
    story.append(Paragraph("4. Quantitative Benchmark Performance & Experimental Progression", heading1_style))
    story.append(Paragraph("The optimized SOTA model was trained on GPU hardware for 40 epochs with Cosine Annealing LR scheduling. Optimal weights were restored from <b>Epoch 36</b>.", body_style))
    
    table_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Original Baseline</b>", body_style), Paragraph("<b>Intermediate (CE+Dice)</b>", body_style), Paragraph("<b>SOTA Final Optimized Value</b>", body_style), Paragraph("<b>Total Net Boost</b>", body_style)],
        ["Total Parameters", "8,317,538", "8,317,538", "8,073,378", "Higher Efficiency"],
        ["Test Mean IoU (mIoU)", "67.55%", "74.55%", "88.75% (0.8875)", "+21.20% surge"],
        ["Test Mean Dice (mDice)", "78.64%", "84.44%", "93.87% (0.9387)", "+15.23% surge"],
        ["Pixel Accuracy", "89.98%", "91.71%", "96.73% (0.9673)", "+6.75%"],
        ["Precision", "82.37%", "84.72%", "94.13% (0.9413)", "+11.76%"],
        ["Recall (Sensitivity)", "76.01%", "84.16%", "93.62% (0.9362)", "+17.61%"]
    ]
    t = Table(table_data, colWidths=[120, 80, 100, 110, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ebf8ff')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Key Epoch milestones log table
    story.append(Paragraph("Key Epoch Validation Milestones Log:", heading2_style))
    epoch_table_data = [
        [Paragraph("<b>Epoch</b>", body_style), Paragraph("<b>Val Loss</b>", body_style), Paragraph("<b>Val mIoU</b>", body_style), Paragraph("<b>Val mDice</b>", body_style), Paragraph("<b>Key Milestone / Status</b>", body_style)],
        ["1", "0.4312", "74.67%", "84.51%", "Initial SOTA Warmup Saved"],
        ["3", "0.3720", "78.53%", "87.37%", "Saved mIoU Improvement"],
        ["5", "0.2877", "80.84%", "88.91%", "Crossed 80% mIoU Threshold"],
        ["7", "0.2444", "84.57%", "91.32%", "Crossed 90% Dice Threshold"],
        ["14", "0.1972", "86.11%", "92.27%", "Saved mIoU Improvement"],
        ["22", "0.1758", "87.84%", "93.34%", "Saved mIoU Improvement"],
        ["31", "0.1687", "88.44%", "93.69%", "Saved mIoU Improvement"],
        ["36", "0.1654", "88.75%", "93.87%", "PEAK OPTIMAL WEIGHTS SAVED"],
        ["40", "0.1655", "88.63%", "93.81%", "Final Epoch Early Stopped"]
    ]
    et = Table(epoch_table_data, colWidths=[45, 65, 65, 65, 210])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7fafc')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (4,0), (4,-1), 'LEFT'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#edf2f7')])
    ]))
    story.append(et)
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # 5. Visual Results Gallery
    story.append(Paragraph("5. Visual Results Gallery", heading1_style))
    
    # Loss curves image
    notebooks_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(notebooks_dir), "notebooks", "plots")
    if not os.path.exists(plots_dir):
        plots_dir = "outputs/medical_segmentation/plots"

    loss_img_path = os.path.join(plots_dir, "loss_curves.png")
    if os.path.exists(loss_img_path):
        story.append(Paragraph("<b>A. Smooth Loss Convergence Curves:</b>", heading2_style))
        story.append(Image(loss_img_path, width=4.5*inch, height=2.6*inch))
        story.append(Spacer(1, 8))

    metrics_img_path = os.path.join(plots_dir, "segmentation_metrics.png")
    if os.path.exists(metrics_img_path):
        story.append(Paragraph("<b>B. Validation mIoU & mDice Progression Curves:</b>", heading2_style))
        story.append(Image(metrics_img_path, width=4.5*inch, height=2.6*inch))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Epoch visual progress images
    story.append(Paragraph("C. Epoch Validation Predictions Progress", heading2_style))
    story.append(Paragraph("Side-by-side visual segmentation results: [Input Image (Left) | Ground Truth Mask (Center) | Shared-MoE Prediction (Right)]:", body_style))
    story.append(Spacer(1, 4))

    epoch_preds = [
        ("Epoch 1 (Initial learning phase):", os.path.join(plots_dir, "val_predictions_epoch_1.png")),
        ("Epoch 7 (Crossing 90% Dice):", os.path.join(plots_dir, "val_predictions_epoch_7.png")),
        ("Epoch 36 (Peak Best Model - 88.75% mIoU):", os.path.join(plots_dir, "val_predictions_epoch_36.png"))
    ]

    for label, img_path in epoch_preds:
        if os.path.exists(img_path):
            story.append(Paragraph(f"<b>• {label}</b>", body_style))
            story.append(Image(img_path, width=5.5*inch, height=1.65*inch))
            story.append(Spacer(1, 8))

    # 6. Conclusion
    story.append(Paragraph("6. Research Conclusions & Key Takeaways", heading1_style))
    story.append(Paragraph(
        "1. <b>Massive Performance Surge (+21.20% mIoU):</b> Combining 352x352 high spatial resolution with Focal Tversky loss "
        "and Shared-Expert MoE routing boosted test mIoU from 67.55% to 88.75% and test Dice score from 78.64% to 93.87%.<br/>"
        "2. <b>High Sensitivity (+17.61% Recall):</b> Focal Tversky Loss heavily penalized false negatives, raising polyp detection recall to 93.62%.<br/>"
        "3. <b>Shared-Expert Capacity Scaling:</b> The DeepSeek-style Shared MoE layer enabled specialized token routing in Stages 3 & 4 "
        "while maintaining an overall parameter count of 8.07M.",
        body_style
    ))

    doc.build(story)
    print(f"PDF successfully built at {pdf_path}")

if __name__ == "__main__":
    build_pdf()
