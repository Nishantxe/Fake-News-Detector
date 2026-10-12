/*
 * engine.js - runs the trained model in the browser.
 * It repeats, step by step, what scikit-learn does in Python:
 *   clean text -> split into words -> drop stop words -> make single words + word pairs
 *   -> TF-IDF scores -> L2 normalise -> weighted sum (the model) -> verdict
 */
(function (root) {
  "use strict";

  // Same cleaning as clean_text() in fake_news_detector_pa.py
  function clean(text) {
    return String(text).toLowerCase()
      .replace(/http\S+|www\.\S+/g, " ")
      .replace(/[^a-z\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function createEngine(model) {
    const index = new Map(model.terms.map(function (t, i) { return [t, i]; }));
    const stop = new Set(model.stop);

    function analyze(text) {
      // Keep the original words (and spaces) so the page can highlight them later.
      const segs = (String(text).match(/\s+|\S+/g) || []).map(function (t) {
        return { text: t, ws: /^\s/.test(t), contrib: 0 };
      });

      // Tokens the model sees: words of 2+ letters that are not stop words.
      const flat = [];
      segs.forEach(function (s, i) {
        if (s.ws) return;
        clean(s.text).split(" ").forEach(function (tok) {
          if (tok.length >= 2 && !stop.has(tok)) flat.push({ tok: tok, owner: i });
        });
      });

      // Count every single word and word pair that exists in the model's vocabulary.
      const found = new Map();
      let knownWords = 0;
      function add(term, owners, isWord) {
        const idx = index.get(term);
        if (idx === undefined) return;
        if (isWord) knownWords++;
        if (!found.has(idx)) found.set(idx, []);
        found.get(idx).push(owners);
      }
      flat.forEach(function (f, j) {
        add(f.tok, [f.owner], true);
        if (j + 1 < flat.length) add(f.tok + " " + flat[j + 1].tok, [f.owner, flat[j + 1].owner], false);
      });

      // TF-IDF vector, then L2 normalisation.
      let sumsq = 0;
      found.forEach(function (list, idx) {
        const x = list.length * model.idf[idx];
        sumsq += x * x;
      });
      if (sumsq === 0) {
        return { known: 0, knownWords: 0, segs: segs, pa: null, lr: null };
      }
      const norm = Math.sqrt(sumsq);

      let paScore = model.pa.b, lrScore = model.lr.b;
      found.forEach(function (list, idx) {
        const unit = model.idf[idx] / norm;          // TF-IDF share of ONE occurrence
        paScore += list.length * unit * model.pa.w[idx];
        lrScore += list.length * unit * model.lr.w[idx];
        const c = unit * model.pa.w[idx];            // what one occurrence pushed the verdict by
        list.forEach(function (owners) {
          owners.forEach(function (o) { segs[o].contrib += c / owners.length; });
        });
      });

      return {
        known: found.size,
        knownWords: knownWords,
        segs: segs,
        // Passive Aggressive has no true probability: squash its score into 0..1 (same as the Python script).
        pa: { score: paScore, conf: sigmoid(2 * paScore) },
        lr: { score: lrScore, conf: sigmoid(lrScore) }
      };
    }

    return { analyze: analyze };
  }

  const api = { createEngine: createEngine, clean: clean };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FND = api;
})(typeof window !== "undefined" ? window : globalThis);
