# Fake News Detector

Paste a headline and see which words make a machine learning model call it fake or real.

- **Model:** TF-IDF + Passive Aggressive Classifier (main), Logistic Regression (baseline)
- **Website:** plain HTML, CSS and JavaScript. The trained model runs **in the browser**, so it works on free GitHub Pages hosting (GitHub Pages cannot run Python).
- **Privacy:** nothing you type leaves your device.

## What is in this repo

| File | What it is |
|---|---|
| `index.html`, `assets/style.css`, `assets/app.js` | The website |
| `assets/engine.js` | Runs the model in the browser (same steps as scikit-learn) |
| `assets/model.js` | The trained model (vocabulary, IDF scores, word weights). Generated, do not edit by hand |
| `export_model.py` | Trains the model and writes `assets/model.js` |
| `fake_news_detector_pa.py` | The full Python project: training, evaluation, confusion matrix, interactive prompt |

## 1. Try it on your computer

Open `index.html` in a browser. That is all, no server needed.

## 2. Retrain on real data (do this before you show it to anyone)

The model that ships in the repo was trained on 60 demo headlines, so it knows few words. For realistic results:

1. Download the **Fake and Real News Dataset** from Kaggle. You get `Fake.csv` and `True.csv`. Put them in this folder.
2. Install and run:
   ```
   pip install -r requirements.txt
   python export_model.py --fake Fake.csv --real True.csv
   ```
3. This overwrites `assets/model.js`. The page automatically shows the new accuracy numbers.

Notes:
- `--max-features 20000` (default) keeps `model.js` around 1 MB so the page loads fast. Raise it for more accuracy, lower it for a smaller file.
- Accuracy on this dataset can look inflated because real articles often name the news agency. Test on a second dataset such as LIAR before you trust it.

## 3. Put it on GitHub

1. Open `index.html` and `README.md` and replace every `YOUR_USERNAME` with your GitHub username (there are two links in `index.html`).
2. Create an empty repository on github.com named `fake-news-detector` (no README), then:
   ```
   git init
   git add .
   git commit -m "Fake news detector website"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
   git push -u origin main
   ```

## 4. Publish with GitHub Pages

1. In the repository, go to **Settings > Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**, pick `main` and the `/ (root)` folder, then Save.
3. After a minute your site is live at `https://YOUR_USERNAME.github.io/fake-news-detector/`.

## 5. Use your own domain

1. In **Settings > Pages > Custom domain**, type your domain (for example `yourname.dev` or `detector.yourname.dev`) and Save. Do this **before** changing DNS.
2. At your domain registrar (GoDaddy, Namecheap, Hostinger and so on), open the DNS settings:

   **For a subdomain** such as `detector.yourname.dev`:

   | Type | Name | Value |
   |---|---|---|
   | CNAME | `detector` | `YOUR_USERNAME.github.io` |

   **For the main domain** such as `yourname.dev`:

   | Type | Name | Value |
   |---|---|---|
   | A | `@` | `185.199.108.153` |
   | A | `@` | `185.199.109.153` |
   | A | `@` | `185.199.110.153` |
   | A | `@` | `185.199.111.153` |
   | CNAME | `www` | `YOUR_USERNAME.github.io` |

   Check GitHub's current docs ("Managing a custom domain for your GitHub Pages site") in case these IP addresses have changed.
3. Wait from a few minutes up to a day for DNS to update. Then tick **Enforce HTTPS** in **Settings > Pages**.

Saving the custom domain in step 1 makes GitHub add a `CNAME` file to your repo. Run `git pull` before your next push so you keep it. (`CNAME.example` is only a reminder of the format.)

## Limitations

- It reads writing style, not facts.
- It only knows words from its training data.
- English only.
- The confidence number is a score, not a true probability.
