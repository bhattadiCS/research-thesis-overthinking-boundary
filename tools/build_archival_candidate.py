"""Prepare the reviewed draft as PDF/A-2b; require veraPDF and unchanged pages.

Run in an isolated authoring environment with pikepdf[pdfa]==10.16.0 and
PyMuPDF. This produces an archival *candidate*, not academic approval or ETD
acceptance. A changed thesis requires a newly reviewed source SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pikepdf
from pikepdf import pdfa
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'output/pdf/Masters_Thesis_Draft_v1_Aditya_Bhatt.pdf'
OUTPUT = ROOT / 'output/pdf/Masters_Thesis_Archival_Candidate_v1_Aditya_Bhatt.pdf'
RECEIPTS = ROOT / 'ThesisDocs/archival'
RENDER = ROOT / 'tmp/pdfs/archival_candidate'
REVIEWED_SOURCE_SHA = '833651e74486ff986e52466b758d4eddf686589b2409badfea41f1f079b45e27'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_front_matter() -> Path:
    """Correct known draft omissions without changing the scientific body."""
    prepared = RENDER / 'front_matter_prepared.pdf'
    overlay_path = RENDER / 'front_matter_overlay.pdf'
    pdfmetrics.registerFont(TTFont('ArchivalArial', 'C:/Windows/Fonts/arial.ttf'))
    overlay = canvas.Canvas(str(overlay_path), pagesize=(612, 792), initialFontName='ArchivalArial')
    overlay.setFont('ArchivalArial', 12)
    overlay.drawCentredString(306, 568, 'Aditya Bhatt')
    overlay.showPage()
    overlay.setFont('ArchivalArial', 12)
    overlay.drawString(72, 792-578, 'Research adviser: Dr. Zerotti Woods')
    overlay.drawString(72, 792-602, 'Second reader: Dr. Moustapha Pemy')
    overlay.setFont('ArchivalArial', 10)
    overlay.drawString(72, 792-645, 'Draft archival candidate. Academic approval, the final submission month and')
    overlay.drawString(72, 792-663, 'institutional acceptance are pending.')
    overlay.showPage()
    with fitz.open(SOURCE) as document:
        cover = document[0]
        name_block = [block for block in cover.get_text('blocks') if block[4].strip() == 'Aditya Bhatt']
        assert len(name_block) == 1
        cover.add_redact_annot(fitz.Rect(name_block[0][:4]) + (-1, -1, 1, 1), fill=(1, 1, 1))
        cover.apply_redactions()
        abstract = document[1]
        abstract.add_redact_annot(fitz.Rect(70, 580, 541, 614), fill=(1, 1, 1))
        abstract.apply_redactions()
        contents = document[2]
        entries = [('Abstract', 'ii')]
        for block in contents.get_text('blocks'):
            if 110 <= block[1] < 500:
                lines = block[4].strip().splitlines()
                assert len(lines) == 2
                entries.append(tuple(lines))
        assert len(entries) == 12
        contents.add_redact_annot(fitz.Rect(70, 110, 541, 500), fill=(1, 1, 1))
        contents.apply_redactions()
        overlay.setFont('ArchivalArial', 11)
        for index, (title, number) in enumerate(entries):
            baseline = 122 + 36 * index
            overlay.drawString(72, 792-baseline, title)
            overlay.drawRightString(540, 792-baseline, number)
        overlay.save()
        with fitz.open(overlay_path) as overlay_pdf:
            for index in range(3):
                document[index].show_pdf_page(document[index].rect, overlay_pdf, index)
        document.save(prepared)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verapdf-jar', required=True, type=Path)
    args = parser.parse_args()
    assert pikepdf.__version__ == '10.16.0', 'Use the recorded isolated authoring dependency'
    assert sha(SOURCE) == REVIEWED_SOURCE_SHA, 'Source differs from the visually reviewed draft'
    jar = args.verapdf_jar.resolve()
    assert jar.is_file()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    RENDER.mkdir(parents=True, exist_ok=True)
    prepared_source = prepare_front_matter()
    repairs = [{'page': 1, 'repair': 'single-space author name below by'},
               {'page': 2, 'repair': 'add adviser/reader names after abstract; relocate draft-status note'},
               {'page': 3, 'repair': 'add Abstract ii to reconstructed contents; preserve chapter/appendix page numbers'}]
    with pikepdf.open(prepared_source) as pdf:
        # The five front-matter pages contain empty transition dictionaries.
        # Removing these has no page appearance effect and avoids an unsupported
        # pikepdf construct. Nonempty transitions are intentionally rejected.
        for i, page in enumerate(pdf.pages, 1):
            if '/Trans' in page.obj:
                if len(page.obj['/Trans']):
                    raise ValueError('Nonempty page transition requires separate review')
                del page.obj['/Trans']
                repairs.append({'page': i, 'repair': 'remove empty /Trans dictionary'})
        for obj in pdf.objects:
            if (isinstance(obj, pikepdf.Dictionary)
                    and obj.get('/Subtype') == pikepdf.Name('/CIDFontType2')
                    and '/CIDToGIDMap' not in obj):
                # The reviewed source has exactly one such embedded Arial font.
                # Final pixel comparison guards every rendered glyph after the
                # explicit map is added. Do not generalize this repair to new PDFs.
                assert str(obj.get('/BaseFont')) == '/ArialMT'
                assert '/FontFile2' in obj['/FontDescriptor']
                obj['/CIDToGIDMap'] = pikepdf.Name('/Identity')
                repairs.append({'object': list(obj.objgen), 'font': str(obj['/BaseFont']),
                                'repair': 'add explicit /Identity CIDToGIDMap'})
        report = pdfa.save(pdf, OUTPUT, '2b')
        assert report.passed
        pike_summary = report.summary()
        prepared = list(report.prepared.describe())
    output_sha = sha(OUTPUT)
    command = ['java', '-jar', str(jar), '--flavour', '2b', '--format', 'xml', str(OUTPUT)]
    validation = subprocess.run(command, capture_output=True, check=False)
    xml_path = RECEIPTS / 'candidate_verapdf_2b.xml'
    xml_path.write_bytes(validation.stdout)
    (RENDER / 'verapdf_stderr.txt').write_bytes(validation.stderr)
    tree = ET.fromstring(validation.stdout)
    reports = list(tree.iter('validationReport'))
    details = list(tree.iter('details'))
    assert validation.returncode == 0, 'veraPDF did not pass; inspect validation XML'
    assert len(reports) == len(details) == 1
    assert reports[0].get('isCompliant') == 'true'
    assert details[0].get('failedRules') == details[0].get('failedChecks') == '0'
    with fitz.open(SOURCE) as original, fitz.open(OUTPUT) as candidate:
        assert len(original) == len(candidate) == 86
        pages = []
        for i, (before, after) in enumerate(zip(original, candidate), 1):
            assert before.rect == after.rect, f'page size changed: {i}'
            edited = i in (1, 2, 3)
            if not edited:
                assert before.get_text() == after.get_text(), f'extracted text changed: {i}'
                assert before.get_text('words') == after.get_text('words'), f'word placement changed: {i}'
            elif i == 1:
                assert sorted(before.get_text().split()) == sorted(after.get_text().split())
            elif i == 2:
                clip = fitz.Rect(0, 0, 612, 560)
                assert before.get_text('words', clip=clip) == after.get_text('words', clip=clip)
                assert 'Research adviser: Dr. Zerotti Woods' in after.get_text()
                assert 'Second reader: Dr. Moustapha Pemy' in after.get_text()
            else:
                old = [block[4].strip() for block in before.get_text('blocks') if 110 <= block[1] < 500]
                assert all(entry in after.get_text().strip() for entry in old)
                assert 'Abstract\nii' in after.get_text()
            a = before.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
            b = after.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
            assert (a.width, a.height, a.n) == (b.width, b.height, b.n)
            delta = np.abs(np.frombuffer(a.samples, dtype=np.uint8).astype(np.int16)
                           - np.frombuffer(b.samples, dtype=np.uint8).astype(np.int16))
            changed = int(np.count_nonzero(np.any(delta.reshape(a.height, a.width, 3), axis=2)))
            maximum = int(delta.max())
            fraction = changed / (a.width * a.height)
            # The ICC output intent introduces one-level gray rounding on some
            # links and table fills. Reject larger differences or geometry shifts.
            # These renders must not be described as byte/pixel identical.
            if not edited:
                assert maximum <= 1 and fraction < .02, f'appearance changed: {i}'
            b.save(RENDER / f'page_{i:03d}.png')
            pages.append({'page': i, 'width': b.width, 'height': b.height,
                          'rgb_pixels_sha256': hashlib.sha256(b.samples).hexdigest(),
                          'intentional_front_matter_edit': edited,
                          'pixels_identical': changed == 0, 'text_identical': not edited,
                          'word_geometry_identical': not edited, 'changed_pixels': changed,
                          'changed_pixel_fraction': fraction, 'max_channel_difference': maximum})
        assert original.get_toc() == candidate.get_toc(), 'Outline changed'
        fonts = {font[0] for page in candidate for font in page.get_fonts(full=True)}
        assert all(candidate.extract_font(xref)[3] for xref in fonts), 'Unembedded font'
    assert sha(SOURCE) == REVIEWED_SOURCE_SHA and sha(OUTPUT) == output_sha
    record = {
        'schema': 'thesis-archival-candidate-v1', 'profile': 'PDF/A-2b',
        'status': 'validated_candidate', 'academic_approval': 'pending',
        'deposit_route': 'EP program confirmation pending; no deposit performed',
        'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': REVIEWED_SOURCE_SHA,
        'output': OUTPUT.relative_to(ROOT).as_posix(), 'sha256': output_sha,
        'bytes': OUTPUT.stat().st_size, 'pages': 86,
        'builder': Path(__file__).relative_to(ROOT).as_posix(), 'builder_sha256': sha(Path(__file__)),
        'pikepdf_version': pikepdf.__version__, 'pikepdf_report': pike_summary,
        'manual_structural_repairs': repairs, 'pikepdf_preparation': prepared,
        'authoritative_validator': 'veraPDF Greenfield 1.30.2',
        'validator_jar_sha256': sha(jar), 'validator_exit_code': validation.returncode,
        'validator_report': xml_path.relative_to(ROOT).as_posix(), 'validator_report_sha256': sha(xml_path),
        'validator_details': details[0].attrib, 'validator_report_attributes': reports[0].attrib,
        'embedded_font_objects': len(fonts), 'render_engine': f'PyMuPDF {fitz.VersionBind}',
        'comparison_dpi': 144, 'unchanged_text_and_word_geometry_pages': 83,
        'scientific_abstract_text_and_word_geometry_identical': True,
        'intentional_front_matter_edit_pages': [1, 2, 3],
        'pixel_identical_pages': sum(page['pixels_identical'] for page in pages),
        'unedited_pages_pixel_max_channel_difference': max(page['max_channel_difference'] for page in pages if not page['intentional_front_matter_edit']),
        'unedited_pages_pixel_max_changed_fraction': max(page['changed_pixel_fraction'] for page in pages if not page['intentional_front_matter_edit']),
        'outline_identical': True, 'page_comparisons': pages,
        'limitations': ['This is the reviewed draft, not a final committee-approved thesis.',
                        'PDF/A-2b conformance does not establish accessibility/tagging or institutional acceptance.',
                        'The EP program must confirm its deposit route, deadlines and required archival profile.'],
    }
    (RECEIPTS / 'candidate_build_manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: record[key] for key in ['status','profile','sha256','pages','validator_details','unchanged_text_and_word_geometry_pages','intentional_front_matter_edit_pages','pixel_identical_pages','unedited_pages_pixel_max_channel_difference']}, indent=2))


if __name__ == '__main__':
    main()
