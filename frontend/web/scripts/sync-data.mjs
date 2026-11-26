// Copies the pipeline's JSON (../data/processed) into web/src/data, where the site imports it.
// Runs before `npm run dev` and `npm run build`. Commit web/src/data: that's what Vercel builds from.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const webDir = path.resolve(import.meta.dirname, "..");
const source = path.resolve(webDir, "../data/processed");
const target = path.join(webDir, "src", "data");
const FILES = ["overview", "model"];

mkdirSync(target, { recursive: true });
if (!existsSync(path.join(source, "overview.json"))) {
  console.log("sync-data: ../data/processed not found, using the committed files in src/data");
} else {
  for (const f of FILES) writeFileSync(path.join(target, `${f}.json`), readFileSync(path.join(source, `${f}.json`)));
  console.log("sync-data: copied data/processed into src/data");
}
