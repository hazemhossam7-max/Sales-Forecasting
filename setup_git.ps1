# PowerShell script to initialize Git and prepare for GitHub upload

Write-Host "=== Sales Forecasting Project - Git Setup ===" -ForegroundColor Green
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed. Please install Git from https://git-scm.com/downloads" -ForegroundColor Red
    exit 1
}

# Initialize git repository if not already initialized
if (Test-Path .git) {
    Write-Host "Git repository already initialized." -ForegroundColor Yellow
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Cyan
    git init
    Write-Host "Git repository initialized successfully!" -ForegroundColor Green
}

# Add all files
Write-Host ""
Write-Host "Adding files to Git..." -ForegroundColor Cyan
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host ""
    Write-Host "Files staged for commit:" -ForegroundColor Cyan
    git status --short
    
    Write-Host ""
    $commitMessage = "Initial commit: Sales Forecasting Project with deployment setup"
    Write-Host "Committing files with message: $commitMessage" -ForegroundColor Cyan
    git commit -m $commitMessage
    Write-Host "Files committed successfully!" -ForegroundColor Green
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

# Check if remote is already set
$remote = git remote get-url origin 2>$null
if ($remote) {
    Write-Host ""
    Write-Host "Remote repository is already set to: $remote" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "=== Next Steps ===" -ForegroundColor Green
    Write-Host "1. Create a new repository on GitHub (don't initialize with README)" -ForegroundColor White
    Write-Host "2. Copy the repository URL (e.g., https://github.com/username/repo-name.git)" -ForegroundColor White
    Write-Host "3. Run the following commands:" -ForegroundColor White
    Write-Host ""
    Write-Host "   git remote add origin YOUR_REPO_URL" -ForegroundColor Cyan
    Write-Host "   git branch -M main" -ForegroundColor Cyan
    Write-Host "   git push -u origin main" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To push to GitHub, run:" -ForegroundColor Yellow
Write-Host "  git remote add origin YOUR_GITHUB_REPO_URL" -ForegroundColor Cyan
Write-Host "  git branch -M main" -ForegroundColor Cyan
Write-Host "  git push -u origin main" -ForegroundColor Cyan

