import re, pathlib
ARCH = '''<figure><svg viewBox="0 0 772 200" xmlns="http://www.w3.org/2000/svg"
 font-family="Carlito,Calibri,Arial,sans-serif">
<defs>
 <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
   orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4a4945"/></marker>
 <marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
   orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#eb6834"/></marker>
</defs>
<g stroke="#4a4945" stroke-width="1.4" fill="none" marker-end="url(#a)">
 <path d="M34,120 H82"/><path d="M216,120 H272"/><path d="M452,120 H508"/>
 <path d="M148,104 V58 H636 V104" stroke="#2a78d6"/>
 <path d="M572,104 V78 H636" stroke="#2a78d6" marker-end="none"/>
</g>
<path d="M572,140 V172 H636" stroke="#eb6834" stroke-width="1.4" fill="none" marker-end="url(#ao)"/>
<text x="4" y="124" font-size="11" fill="#141412">item</text>
<g stroke="#141412" stroke-width="1.2" fill="#f8f8f6">
 <rect x="84" y="100" width="130" height="40" rx="3"/>
 <rect x="274" y="100" width="176" height="40" rx="3"/>
 <rect x="510" y="100" width="124" height="40" rx="3"/>
</g>
<g font-size="11" font-weight="700" fill="#141412">
 <text x="94" y="117">S1 &middot; DistilBERT</text>
 <text x="284" y="117">S2 &middot; retrieve + rerank</text>
 <text x="520" y="117">S3 &middot; LLM agent</text>
</g>
<g font-size="9" fill="#82817c">
 <text x="94" y="131">69 head classes + OTHER</text>
 <text x="284" y="131">530 label documents &rarr; top-10</text>
 <text x="520" y="131">forced tool call</text>
</g>
<g font-size="9.5" fill="#2a78d6" font-weight="700">
 <text x="156" y="52">83.5% answered here &mdash; accuracy 0.977</text>
 <text x="646" y="82">label</text>
</g>
<text x="646" y="176" font-size="9.5" fill="#eb6834" font-weight="700">abstain &rarr; human</text>
<g font-size="9.5" fill="#eb6834" font-weight="700"><text x="150" y="160">16.5% deferred</text></g>
<text x="150" y="173" font-size="9" fill="#82817c">argmax == OTHER  or  p_max &lt; 0.900</text>
<path d="M148,140 V152" stroke="#eb6834" stroke-width="1.4" fill="none"/>
</svg><figcaption>Each tier answers what it can. Cost and capability rise only for the
items that need them &mdash; and an abstention is a refusal to label, not a fallback guess.</figcaption></figure>'''

def fig(path, cap):
    svg = pathlib.Path(path).read_text()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg width="[^"]*" height="[^"]*"', '<svg', svg, count=1)
    return f'<figure>{svg}<figcaption>{cap}</figcaption></figure>'

html = pathlib.Path("portfolio.html").read_text()
html = html.replace("{{ARCH}}", ARCH)
html = html.replace("{{FRONTIER}}", fig("fig_frontier.svg",
  "Accuracy against the deferral threshold, replayed from cached Stage&nbsp;3 answers at zero "
  "API cost. Micro falls as macro rises; zero-shot macro nearly doubles. The curve is still "
  "climbing where the system operates."))
html = html.replace("{{LATENCY}}", fig("fig_latency.svg",
  "200 served requests against a live LLM provider. Cold start excluded and reported "
  "separately: the retriever is lazy, so the first deferred request after a restart pays 10.8 s "
  "to load it &mdash; which 82% of requests never do."))
pathlib.Path("portfolio.final.html").write_text(html)
print("ok", len(html), "bytes;", "unresolved:", [t for t in ("{{ARCH}}","{{FRONTIER}}","{{LATENCY}}") if t in html])
