/* app.js - connects the page to the model (see engine.js for the maths). */
(function () {
  "use strict";

  var M = window.FND_MODEL;
  var input = document.getElementById("input");
  var result = document.getElementById("result");
  var note = document.getElementById("model-note");
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!M) {
    result.hidden = false;
    result.textContent = "The model file (assets/model.js) did not load. Check that it is in the repository.";
    return;
  }
  var engine = window.FND.createEngine(M);

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function pct(x) { return Math.round(x * 100) + "%"; }

  // ----- small note under the checker: where the model came from -----
  (function renderNote() {
    var m = M.meta, unit = m.source === "demo" ? "sample headlines" : "articles";
    var t = "This model learned from " + m.n_train.toLocaleString() + " " + unit + ". On " +
      m.n_test.toLocaleString() + " unseen texts, Passive Aggressive got " + pct(m.metrics.pa.accuracy) +
      " right and Logistic Regression " + pct(m.metrics.lr.accuracy) + ".";
    if (m.source === "demo") {
      t += " That is a small demo model, so it knows few words. Retrain it on the Kaggle dataset for realistic results (steps are in the README).";
    }
    note.textContent = t;
  })();

  function showMessage(title, body) {
    result.replaceChildren(el("p", "msg-title", title), el("p", "msg-body", body));
    result.hidden = false;
  }

  function topWords(segs, sign) {
    var best = new Map();
    segs.forEach(function (s) {
      if (s.ws) return;
      var w = s.text.toLowerCase().replace(/[^a-z]/g, "");
      if (w.length < 2 || s.contrib * sign <= 0) return;
      best.set(w, (best.get(w) || 0) + Math.abs(s.contrib));
    });
    return Array.from(best.entries()).sort(function (a, b) { return b[1] - a[1]; })
      .slice(0, 4).map(function (e) { return e[0]; });
  }

  function render(a, original) {
    if (!a.known) {
      result.replaceChildren(
        el("p", "plain", original),
        el("p", "msg-title", "Can\u2019t judge this one"),
        el("p", "msg-body", "None of these words are in the model\u2019s vocabulary. Try a longer headline, or retrain the model on more data.")
      );
      result.hidden = false;
      return;
    }

    var isFake = a.pa.conf >= 0.5;
    var frag = document.createDocumentFragment();

    // 1. verdict
    var v = el("p", "verdict " + (isFake ? "fake" : "real"));
    v.appendChild(el("strong", null, isFake ? "Looks fake" : "Looks real"));
    v.appendChild(el("span", null, "fake-confidence " + pct(a.pa.conf)));
    frag.appendChild(v);

    // 2. the headline, marked up word by word
    var maxAbs = a.segs.reduce(function (m, s) { return Math.max(m, Math.abs(s.contrib)); }, 0) || 1;
    var p = el("p", "annotated");
    var marks = [];
    a.segs.forEach(function (s) {
      if (s.ws || Math.abs(s.contrib) / maxAbs < 0.12) { p.appendChild(document.createTextNode(s.text)); return; }
      var m = el("span", "mark " + (s.contrib > 0 ? "fake" : "real"), s.text);
      m.style.setProperty("--a", (0.3 + 0.45 * Math.abs(s.contrib) / maxAbs).toFixed(2));
      p.appendChild(m); marks.push(m);
    });
    frag.appendChild(p);

    // 3. meter
    var meter = el("div", "meter");
    meter.setAttribute("role", "img");
    meter.setAttribute("aria-label", "Fake-confidence " + pct(a.pa.conf));
    var track = el("div", "meter-track"), pin = el("span", "meter-pin");
    track.appendChild(pin); meter.appendChild(track);
    var ends = el("div", "meter-ends"); ends.appendChild(el("span", null, "Real")); ends.appendChild(el("span", null, "Fake"));
    meter.appendChild(ends); frag.appendChild(meter);

    // 4. strongest words
    var sig = el("div", "signals");
    var tf = topWords(a.segs, 1), tr = topWords(a.segs, -1);
    if (tf.length) { var l1 = el("p"); l1.appendChild(el("span", "toward-fake", "Pushes toward fake: ")); l1.appendChild(document.createTextNode(tf.join(", "))); sig.appendChild(l1); }
    if (tr.length) { var l2 = el("p"); l2.appendChild(el("span", "toward-real", "Pushes toward real: ")); l2.appendChild(document.createTextNode(tr.join(", "))); sig.appendChild(l2); }
    frag.appendChild(sig);
    frag.appendChild(el("p", "legend", "Red marks push toward fake, teal marks push toward real. Darker means a bigger push."));

    // 5. both models
    var dl = el("dl", "models");
    dl.appendChild(el("dt", null, "Passive Aggressive")); dl.appendChild(el("dd", null, (a.pa.conf >= 0.5 ? "fake" : "real") + ", fake-confidence " + pct(a.pa.conf)));
    dl.appendChild(el("dt", null, "Logistic Regression")); dl.appendChild(el("dd", null, (a.lr.conf >= 0.5 ? "fake" : "real") + ", fake-probability " + pct(a.lr.conf)));
    frag.appendChild(dl);

    var ev = "Based on " + a.knownWords + (a.knownWords === 1 ? " word" : " words") + " the model knows.";
    if (a.knownWords < 3) ev += " That is very little, so treat this as a guess.";
    frag.appendChild(el("p", "evidence", ev));

    result.replaceChildren(frag);
    result.hidden = false;

    // The one animated moment: the marker sweeps across the words, then the pin slides to its place.
    if (reduceMotion) {
      marks.forEach(function (m) { m.classList.add("on"); });
      pin.style.left = (a.pa.conf * 100) + "%";
      return;
    }
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        var step = Math.min(70, 900 / Math.max(marks.length, 1));
        marks.forEach(function (m, i) { m.style.transitionDelay = (i * step) + "ms"; m.classList.add("on"); });
        pin.style.left = (a.pa.conf * 100) + "%";
      });
    });
  }

  function run() {
    var text = input.value.trim();
    if (!text) {
      showMessage("Nothing to check yet", "Type or paste a headline above, then press Check headline.");
      input.focus();
      return;
    }
    render(engine.analyze(text), text);
  }

  document.getElementById("check").addEventListener("click", run);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
  });
  document.querySelectorAll(".chip").forEach(function (b) {
    b.addEventListener("click", function () { input.value = b.getAttribute("data-sample"); run(); });
  });
})();
