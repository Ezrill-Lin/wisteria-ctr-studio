# Frontend Deployment Guide

## GitHub Pages Deployment (Automated)

### One-Time Setup

1. **Enable GitHub Pages** in your repository:
   - Go to: https://github.com/Ezrill-Lin/wisteria-ctr-studio/settings/pages
   - Under "Source", select: **GitHub Actions**
   - Click "Save"

### Automatic Deployment

The frontend automatically deploys when you push changes:

```bash
# Make your changes to frontend code
cd frontend
# ... edit files ...

# Commit and push (from root directory)
cd ..
git add -A
git commit -m "Update frontend"
git push
```

The GitHub Action will:
1. Detect changes in the `frontend/` directory
2. Build the production bundle
3. Deploy to GitHub Pages
4. Complete in ~2-3 minutes

### Manual Deployment Trigger

You can also trigger deployment manually:
1. Go to: https://github.com/Ezrill-Lin/wisteria-ctr-studio/actions
2. Select "Deploy Frontend to GitHub Pages"
3. Click "Run workflow"

### After Deployment

Your app will be available at:
- **Production URL**: https://ezrill-lin.github.io/wisteria-ctr-studio/ (or custom domain if configured)
- **Backend API**: https://wisteria-ctr-studio-azlh47c4pq-uc.a.run.app

### Local Development

To test locally before deploying:

```bash
# Start development server
cd frontend
npm run dev

# Open http://localhost:5173
```

To use the local API (instead of production):
1. Change `API_URL` in both predictor components to `'http://localhost:8080'`
2. Run `python api.py` from the root directory

### Manual Build (Testing)

To test the production build locally:

```bash
cd frontend
npm run build
npm run preview
```
