import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class LatexNumberedCanvas(canvas.Canvas):
    """
    Minimalist LaTeX-style canvas: Running header (pages 2+) and bottom page numbering.
    Pure black and white academic styling.
    """
    def __init__(self, *args, **kwargs):
        super(LatexNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.black)
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "JHU ACM 625.804 Semester 2 Thesis Status Report & Completion Plan")
            self.drawRightString(612 - 54, 750, "Aditya Bhatt")
            self.setStrokeColor(colors.black)
            self.setLineWidth(0.4)
            self.line(54, 743, 612 - 54, 743)
            
        # Footer (all pages) - Simple centered page number in classic LaTeX style
        page_text = f"{self._pageNumber}"
        self.drawCentredString(612 / 2.0, 36, page_text)
        self.restoreState()


def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Classic Academic LaTeX Typographic Styles (Times-Roman, pure monochromatic)
    title_style = ParagraphStyle(
        'LatexTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.black,
        alignment=1, # Centered
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'LatexSubtitle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=11,
        leading=14,
        textColor=colors.black,
        alignment=1, # Centered
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'LatexH1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'LatexH2',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'LatexBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=13.5,
        textColor=colors.black,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'LatexBullet',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3.5
    )
    
    meta_table_cell = ParagraphStyle(
        'MetaCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.black
    )

    meta_table_cell_bold = ParagraphStyle(
        'MetaCellBold',
        parent=meta_table_cell,
        fontName='Times-Bold'
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("Overthinking Boundary in Reasoning LLMs", title_style))
    story.append(Paragraph("Semester 2 Status Report and Completion Plan for MATH ACM 625.804", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceAfter=8))

    # Clean Metadata Block
    meta_data = [
        [
            Paragraph("<b>Student:</b> Aditya Bhatt (abhatt25@jh.edu)", meta_table_cell),
            Paragraph("<b>Course:</b> MATH ACM 625.804 (Semester 2 Thesis)", meta_table_cell)
        ],
        [
            Paragraph("<b>Program:</b> M.S. in Applied & Computational Mathematics", meta_table_cell),
            Paragraph("<b>Term:</b> Fall 2026 (Completion by Dec 2026)", meta_table_cell)
        ],
        [
            Paragraph("<b>Research Adviser:</b> Dr. Zerotti Woods (JHU ACM / APL)", meta_table_cell),
            Paragraph("<b>Second Reader:</b> Dr. Moustapha Pemy (JHU ACM / Towson)", meta_table_cell)
        ],
        [
            Paragraph("<b>Program Manager:</b> Dr. Beryl Castello (beryl.castello@jhu.edu)", meta_table_cell),
            Paragraph("<b>Program Chair:</b> Dr. James Spall (james.spall@jhuapl.edu)", meta_table_cell)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[3.5*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceBefore=4, spaceAfter=8))

    # Section 1: Executive Overview & Problem Motivation
    story.append(Paragraph("1. Problem Motivation & Core Concept", h1_style))
    story.append(Paragraph(
        "Modern Artificial Intelligence models (such as DeepSeek and Qwen) solve complex math and logic problems by generating step-by-step <i>'thoughts'</i> before giving a final answer. "
        "However, if an AI is allowed to think for too long, it frequently runs into two major problems:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Overthinking (Corruption):</b> The AI actually figures out the right answer early on, but keeps second-guessing itself until it changes its answer to a wrong one.<br/>"
        "• <b>Wasted Computing Power:</b> The AI gets stuck in repetitive, unproductive loops that burn expensive computing resources without making progress.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>The Mathematical Solution:</b> We treat this as an <i>Optimal Stopping Problem</i>. The goal is to detect the exact moment the AI has reached its maximum potential for correctness, and stop it before it corrupts its own answer. To do this, we calculate the expected net benefit of generating one more thought step:",
        body_style
    ))

    # Math Box 1
    story.append(Spacer(1, 4))
    math_p1 = Paragraph("<b>Optimal Stopping Formula (Net Marginal Utility):</b><br/><br/><i>Expected Utility (&mu;<sub>t</sub>) &nbsp;=&nbsp; [ Probability of Fixing a Mistake ] &nbsp;-&nbsp; [ Probability of Corrupting a Right Answer ] &nbsp;-&nbsp; [ Step Cost ]</i>", ParagraphStyle('MathCenter', parent=body_style, alignment=1))
    math_table1 = Table([[math_p1]], colWidths=[6.5*inch])
    math_table1.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.8, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9F9F9")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(math_table1)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The system automatically stops the model at the exact boundary step (<b>T*</b>) where continuing yields zero or negative expected benefit (&mu;<sub>t</sub> &le; 0). "
        "To measure these probabilities in real-time without knowing the final answer, we developed <i>'Kinematics of Thought'</i> variables—tracking not just the AI's confidence, but the <i>Velocity</i>, <i>Acceleration</i>, and higher-order physics of its confidence changes.",
        body_style
    ))

    # Math Box 2
    story.append(Spacer(1, 4))
    math_p2 = Paragraph("<b>Kinematics of Thought (Higher-Order Derivatives):</b><br/><br/><i>State Kinematics (K) &nbsp;=&nbsp; [ Confidence, &nbsp;Velocity (v), &nbsp;Acceleration (a), ... ]</i>", ParagraphStyle('MathCenter2', parent=body_style, alignment=1))
    math_table2 = Table([[math_p2]], colWidths=[6.5*inch])
    math_table2.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.8, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9F9F9")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(math_table2)
    story.append(Spacer(1, 8))

    # Section 2: Completed Accomplishments (Semester 1)
    story.append(Paragraph("2. Semester 1 Accomplishments & Key Results", h1_style))
    story.append(Paragraph(
        "During Semester 1, the bulk of the computational heavy-lifting, scientific discovery, and primary data collection was completed. To maintain strict organizational rigor, our progress over the last 3 months was executed in four distinct phases:",
        body_style
    ))
    
    story.append(Paragraph("Phase 1: The Global Model Sweep & Deep Data Collection", h2_style))
    story.append(Paragraph(
        "• <b>Inference Pipeline & Ensemble:</b> Executed a massive 52-cell inference pipeline prompting a highly stratified ensemble of 15 reasoning models (from 0.5B parameters up to 32B parameters). This included leading architectures such as DeepSeek R1 Distill, Qwen 2.5, Llama, Mistral, and Yi to ensure architectural diversity.<br/>"
        "• <b>Variable Isolation Strategy:</b> Rigorously locked all control variables (system prompts, decoding algorithms, and random seeds) to mathematically isolate the causal effect of specific independent variables—namely Model Scale, Step Budget limit, 4-bit Quantization, and Softmax Temperatures (scaled across 0.1, 0.6, and 1.0)—on the final Trajectory Correctness.<br/>"
        "• <b>Advanced Infrastructure:</b> Engineered a resilient, dual-environment research infrastructure scaling from Google Colab L4 instances to advanced RUN:AI pods. We utilized custom autonomous orchestrator scripts equipped with fold-level checkpointing and memory-safe 4-bit GPU offloading to handle over 22 hours of continuous, intensive inference without crashing.<br/>"
        "• <b>Dataset Generation:</b> Captured a staggering 192,640 individual reasoning steps across 28,928 problem-solving traces spanning 4 distinct mathematical benchmarks (GSM8K, MATH, GPQA, and ARC). Every intermediate thought was rigidly graded against the final correctness state, yielding the mathematically verified dataset required for downstream neural network training.",
        bullet_style
    ))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("Phase 2: Architectural Innovations & Hyperparameter Tuning", h2_style))
    story.append(Paragraph(
        "• <b>The Physics of Thought:</b> Treated the AI's internal token confidence as a physical object moving through space to mathematically define the boundary of overthinking. We extracted higher-order kinematic derivatives—Velocity, Acceleration, up to the 5th derivative (Crackle)—smoothing the noisy token data with a customized 16-scale Exponential Moving Average (EMA).<br/>"
        "• <b>Dual Neural Architectures:</b> Deployed advanced, non-linear neural probes capable of handling complex physics dynamics. Specifically, we built and optimized a CausalAttractorGRU alongside a specialized Mixture-of-Experts (MoE) neural architecture.<br/>"
        "• <b>Blackwell GPU Tournament:</b> Executed a massive 5-day continuous hyperparameter tuning tournament utilizing an NVIDIA RTX PRO 6000 Blackwell GPU. We trained 100 PyTorch neural configurations across 1,000 epochs to identify the singular most optimal boundary detector.<br/>"
        "• <b>Statistical Verification:</b> Utilized strict Out-of-Fold (OOF) cross-validation combined with a 10,000-draw statistical bootstrap verification suite to prevent AI overfitting. This rigorous iteration successfully produced a real-time neural detector that predicts the optimal stopping point with <b>97.17% OOF AUC accuracy</b>, ultimately driving a <b>60.2% absolute reduction</b> in live overthinking errors.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    # Clean Results Summary Table
    table_data = [
        [
            Paragraph("<b>Metric Evaluated</b>", meta_table_cell_bold),
            Paragraph("<b>Standard Baseline</b>", meta_table_cell_bold),
            Paragraph("<b>Our Optimal Stopping Framework</b>", meta_table_cell_bold),
            Paragraph("<b>Scientific & Practical Impact</b>", meta_table_cell_bold)
        ],
        [
            Paragraph("Trajectory Correctness", meta_table_cell),
            Paragraph("Degrades by up to 15% in late steps", meta_table_cell),
            Paragraph("<b>Halts precisely at peak correctness</b>", meta_table_cell_bold),
            Paragraph("Prevents correct answers from being corrupted by over-generation", meta_table_cell)
        ],
        [
            Paragraph("Overthinking Drift Errors", meta_table_cell),
            Paragraph("1,168 incorrect final answers", meta_table_cell),
            Paragraph("<b>465 incorrect final answers</b>", meta_table_cell_bold),
            Paragraph("<b>60.2% reduction</b> in overthinking-induced failures", meta_table_cell)
        ],
        [
            Paragraph("Detector Accuracy (OOF AUC)", meta_table_cell),
            Paragraph("No Pre-existing Detector (0%)", meta_table_cell),
            Paragraph("<b>97.17% AUC (Kinematic MoE)</b>", meta_table_cell_bold),
            Paragraph("First highly reliable real-time signal for boundary detection", meta_table_cell)
        ],
        [
            Paragraph("Multi-Model Consensus", meta_table_cell),
            Paragraph("Single-model isolated evaluation", meta_table_cell),
            Paragraph("<b>95.47% AUC (Peer-Dynamics)</b>", meta_table_cell_bold),
            Paragraph("Proves AI models can reliably vote to halt one another", meta_table_cell)
        ],
        [
            Paragraph("Token Compute Efficiency", meta_table_cell),
            Paragraph("Computes until arbitrary limit", meta_table_cell),
            Paragraph("<b>Halts exactly when utility &le; 0</b>", meta_table_cell_bold),
            Paragraph("Massive reduction in API token costs and inference time", meta_table_cell)
        ]
    ]

    res_table = Table(table_data, colWidths=[1.4*inch, 1.4*inch, 1.9*inch, 2.3*inch])
    res_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Spacer(1, 4))
    story.append(res_table)
    story.append(Spacer(1, 8))

    # Add Empirical Plots
    story.append(Paragraph("<b>Empirical Evidence: Overthinking Drift & Optimal Stopping Utility</b>", body_style))
    story.append(Spacer(1, 4))
    
    img_path1 = r"C:\Aditya_Data\Personal\ResearchThesis\ThesisDocs\images\overthinking_drift_by_step.png"
    img_path2 = r"C:\Aditya_Data\Personal\ResearchThesis\ThesisDocs\images\stopping_utility_by_step.png"
    
    try:
        plot1 = Image(img_path1, width=3.3*inch, height=2.2*inch)
        plot2 = Image(img_path2, width=3.3*inch, height=2.2*inch)
        plot_table = Table([[plot1, plot2]], colWidths=[3.4*inch, 3.4*inch])
        plot_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(plot_table)
    except Exception as e:
        print("Could not load images:", e)

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i><b>Figure Analysis:</b> The left plot illustrates <b>Overthinking Drift</b>—demonstrating how an AI's accuracy degrades by up to 15% as it continues generating tokens past its peak correctness. The model attends to its own redundant writing, compounding minor logical slips into final errors. The right plot illustrates <b>Optimal Stopping Utility</b>, proving that our early-halting framework maintains robust, positive economic and accuracy utility across a wide range of computational cost penalties.</i>",
        ParagraphStyle('Caption', parent=body_style, fontSize=9, leading=12)
    ))
    story.append(Spacer(1, 8))

    # Section 3: Accelerated Completion Plan (Semester 2: Sept – Dec 2026)
    story.append(Paragraph("3. Accelerated Completion Plan & Remaining Tasks (Semester 2)", h1_style))
    story.append(Paragraph(
        "To ensure all degree requirements, manuscript writing, public defense proceedings, library ETD archiving, and grade approvals are <b>fully completed by the first week of December 2026</b>, the remaining tasks are front-loaded during September, October, and November.",
        body_style
    ))

    # Clean Monochrome Schedule Table
    sched_data = [
        [
            Paragraph("<b>Timeframe</b>", meta_table_cell_bold),
            Paragraph("<b>Research & Computational Tasks</b>", meta_table_cell_bold),
            Paragraph("<b>Thesis Writing & Administrative Deliverables</b>", meta_table_cell_bold)
        ],
        [
            Paragraph("<b>September 2026</b><br/>(Weeks 1 – 4)", meta_table_cell),
            Paragraph("• More feature hyperparameter tuning and advanced concepts.<br/>• Stress testing the thesis ideas across more variables.", meta_table_cell),
            Paragraph("• Draft Chapter 1 (Introduction).<br/>• Draft Chapter 2 (Mathematical Model).<br/>• Draft Chapter 3 (Experimental Setup).", meta_table_cell)
        ],
        [
            Paragraph("<b>October 2026</b><br/>(Weeks 5 – 8)", meta_table_cell),
            Paragraph("• Explore and test other related ideas/edge cases.<br/>• Finalize publication charts and figures.", meta_table_cell),
            Paragraph("• Draft Chapter 4 (Results) & Chapter 5 (Discussion).<br/>• <b>OCTOBER 23:</b> Submit Full Manuscript Draft v1.0 to Dr. Woods & Dr. Pemy.", meta_table_cell)
        ],
        [
            Paragraph("<b>November 2026</b><br/>(Weeks 9 – 12)", meta_table_cell),
            Paragraph("• Prepare 30-min Defense slide deck.<br/>• Finalize technical paper for journal submission.", meta_table_cell),
            Paragraph("• Nov 1–10: Incorporate advisor feedback.<br/>• Nov 11–20: Conduct mock defense dry runs.<br/>• <b>NOV 23–30:</b> Execute 60-Min Public Oral Defense.", meta_table_cell)
        ],
        [
            Paragraph("<b>December 2026</b><br/>(Week 1)", meta_table_cell),
            Paragraph("• Codebase archival & public repo tagging.<br/>• Submit paper to refereed journal venue.", meta_table_cell),
            Paragraph("• Dec 1–3: Final post-defense manuscript polish.<br/>• Dec 4–5: Electronic Thesis (ETD) submission.<br/>• <b>Dec 6–7:</b> Chair sign-off & grade clearance.", meta_table_cell)
        ]
    ]

    sched_table = Table(sched_data, colWidths=[1.3*inch, 2.8*inch, 2.9*inch])
    sched_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(sched_table)
    story.append(Spacer(1, 8))

    # Section 4: Deliverables & Administrative Clearance
    story.append(Paragraph("4. Final Deliverables & Administrative Clearance", h1_style))
    story.append(Paragraph(
        "Upon completion, the project will yield: (1) An Electronic Master's Thesis submitted to the JHU Sheridan Libraries ETD repository (http://guides.library.jhu.edu/etd); (2) A 60-minute Public Oral Defense (30-min presentation + 30-min Q&A); and (3) A publication-quality technical paper submitted to a peer-reviewed venue certifying >60% primary author contribution. Formal signature approvals are submitted concurrently via the official JHU ACM Semester 2 Approval Form.",
        body_style
    ))

    doc.build(story, canvasmaker=LatexNumberedCanvas)
    print(f"LaTeX-style PDF successfully generated at: {filename}")

if __name__ == '__main__':
    out_pdf = r"C:\Aditya_Data\Personal\ResearchThesis\JHU_ACM_625.804_Thesis_Enrollment_Status_Report_Aditya_Bhatt.pdf"
    build_pdf(out_pdf)
