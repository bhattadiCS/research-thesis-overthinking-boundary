// Render trusted local thesis equations with a pinned local KaTeX install.
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
const [runtime, inputPath, outputPath] = process.argv.slice(2);
if (!runtime || !inputPath || !outputPath) {
  throw new Error('Usage: node tools/render_thesis_math.mjs RUNTIME INPUT.json OUTPUT.json');
}
const require = createRequire(path.resolve(runtime, 'package.json'));
const katex = require('katex');
if (katex.version !== '0.19.0') throw new Error('KaTeX must be exactly 0.19.0');
const equations = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const rendered = equations.map(({tex, display}) => katex.renderToString(tex, {
  displayMode: display, throwOnError: true, strict: 'error', trust: false,
  output: 'htmlAndMathml', maxExpand: 1000, maxSize: 25,
}));
fs.writeFileSync(outputPath, JSON.stringify(rendered));
