from pathlib import Path
from research_assistant import extract_pdf_text


pdf = Path('research_assistant/index_demo/Machine Learning Algorithms.pdf')
pages = extract_pdf_text(pdf)
print('pages:', len(pages))
for i, t in pages:
    print('page', i, 'len', len(t), 'nonspace', sum(1 for ch in t if not ch.isspace()))
    print('preview:', repr(t[:200]))

