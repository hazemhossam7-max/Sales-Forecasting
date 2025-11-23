# Deploying to GitHub - Step by Step Guide

## Prerequisites
1. Install Git if not already installed: https://git-scm.com/downloads
2. Create a GitHub account if you don't have one: https://github.com
3. Create a new repository on GitHub (don't initialize with README)

## Steps to Upload

### Option 1: Using Git Commands (Recommended)

1. **Initialize Git repository** (if not already initialized):
```bash
git init
```

2. **Add all files**:
```bash
git add .
```

3. **Commit the files**:
```bash
git commit -m "Initial commit: Sales Forecasting Project with deployment setup"
```

4. **Add your GitHub repository as remote** (replace with your repo URL):
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

5. **Push to GitHub**:
```bash
git branch -M main
git push -u origin main
```

### Option 2: Using GitHub Desktop
1. Download GitHub Desktop: https://desktop.github.com
2. Open GitHub Desktop
3. File → Add Local Repository
4. Select this project folder
5. Click "Publish repository"
6. Enter repository name and description
7. Click "Publish repository"

### Option 3: Using VS Code
1. Open the project in VS Code
2. Click on the Source Control icon (left sidebar)
3. Click "Initialize Repository"
4. Stage all changes (click + next to files)
5. Enter commit message
6. Click "Commit"
7. Click "..." → "Publish Branch"
8. Follow prompts to create GitHub repository

## After Uploading

1. **Verify files are uploaded**: Check your GitHub repository
2. **Set up Streamlit Cloud** (optional):
   - Go to https://streamlit.io/cloud
   - Connect your GitHub account
   - Select your repository
   - Deploy!

## Important Notes

- The `.gitignore` file ensures sensitive files (models, data, etc.) are not uploaded
- Large data files should be stored separately (GitHub LFS or external storage)
- Update `README.md` with your actual repository URL
- Consider adding a LICENSE file

