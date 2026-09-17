from playwright.sync_api import sync_playwright
import pathlib
src = pathlib.Path("portfolio.final.html").resolve().as_uri()
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page()
    pg.goto(src, wait_until="networkidle")
    pg.pdf(path="Jayden_Lee_Cascade_Classifier.pdf", format="A4",
           print_background=True, prefer_css_page_size=True)
    b.close()
print("pdf written")
