# CS441-Final-Project

Sales Forecasting / Demand Prediction

Goal: Predict the number of sales per menu item for the next hour/day.

## Development environment

We use a local virtual environment stored in `.venv/` for development. A `.gitignore`
entry was added to ignore `.venv/` so the environment itself is not committed.

Activation and installation (macOS / Linux, bash):

1. Activate the venv:

	```bash
	cd /Users/Fabbe/UIUC/CS441/CS441-Final-Project
	source .venv/bin/activate
	```
	```bash
	cd /Users/tobiashuber/Documents/GitHub/CS441-Final-Project
	source .venv/bin/activate
	```
	

2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

To add or update dependencies locally and commit the resulting pinned list:

```bash
# After installing packages inside the activated venv
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update Python dependencies"
```

If you'd like, I can pin specific versions in `requirements.txt` (recommended) or
install these packages into the created `.venv` to verify installation; tell me which you prefer.

3. Run Streamlit

To start the streamlit server:
```bash
streamlit run app.py
```