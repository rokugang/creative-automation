# Direct Commands for Recording

## Option 1: Use the Menu (Interactive)
```bash
python run.py
# Then select 1 for web interface
```

## Option 2: Direct Streamlit Launch (Faster for Recording)
```bash
python -m streamlit run app.py
```

## Option 3: If you're using python3 specifically
```bash
python3 -m streamlit run app.py
```

## Full Recording Command Sequence

```bash
# 1. Show where you are
dir

# 2. Launch the web interface directly
python -m streamlit run app.py

# 3. Browser opens automatically to http://localhost:8501
# In browser:
#   - Click "Browse files"
#   - Select examples/sample_simple.json
#   - Click "Process Campaign" 
#   - Wait ~30 seconds for completion

# 4. After processing, in new terminal window:
explorer outputs\demo\

# 5. Show the generated files in Explorer
```

## If Streamlit Doesn't Launch Automatically

The browser should open automatically, but if not:
1. Look for the URL in terminal (usually http://localhost:8501)
2. Open browser manually
3. Navigate to http://localhost:8501

## Troubleshooting

If you get any errors:
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Specifically install streamlit
pip install streamlit

# Check Python version
python --version
```

## Quick Test Before Recording

Run this to make sure everything works:
```bash
python src/main.py demo
```

This will generate test assets to confirm your API key is working.
