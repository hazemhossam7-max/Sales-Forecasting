# Quick Start Guide

## 🚀 Quick Deployment Steps

### 1. Upload to GitHub

**Option A: Using Command Line**
```powershell
# 1. Create a new repository on GitHub (don't initialize with README)
# 2. Run these commands (replace YOUR_USERNAME and YOUR_REPO_NAME):

git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Option B: Using GitHub Desktop**
1. Download GitHub Desktop: https://desktop.github.com
2. File → Add Local Repository → Select this folder
3. Click "Publish repository"
4. Enter repository name and click "Publish"

### 2. Run Locally

**Windows:**
```powershell
.\run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Or manually:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Run with Docker

```bash
docker build -t sales-forecasting .
docker run -p 8501:8501 sales-forecasting
```

Or with docker-compose:
```bash
docker-compose up
```

### 4. Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Click "Deploy"

## 📁 Project Files Created

- `app.py` - Streamlit web application
- `src/` - Source code modules
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose setup
- `README.md` - Comprehensive documentation
- `.gitignore` - Git ignore rules
- `LICENSE` - MIT License

## 🔧 Next Steps

1. **Update README.md**: Replace placeholder URLs with your actual GitHub repo URL
2. **Add Data**: Upload your CSV files through the web interface
3. **Train Models**: Use the Model Training page to train your models
4. **Make Predictions**: Use the Predictions page to forecast sales

## 📝 Notes

- The notebook file is kept for reference but not required for deployment
- Model files are automatically gitignored (stored in `models/` directory)
- Data files should be uploaded through the web interface or stored externally

