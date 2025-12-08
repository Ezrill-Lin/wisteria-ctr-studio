# Frontend Deployment Guide

## Netlify Deployment (Recommended - Easiest)

### One-Time Setup

1. **Sign up for Netlify**: https://app.netlify.com/signup (free tier)
2. **Connect GitHub**: Click "Add new site" → "Import an existing project"
3. **Select your repository**: Choose `wisteria-ctr-studio`
4. **Configure build settings**:
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/dist`
5. **Click "Deploy"**

### Automatic Deployment

After setup, Netlify automatically deploys when you push to GitHub:
- Detects changes in `frontend/` directory
- Builds and deploys in ~2-3 minutes
- Provides a live URL like: `https://your-app-name.netlify.app`

### Manual Deployment

You can also deploy manually using Netlify CLI:

```bash
# Install Netlify CLI (one-time)
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy from frontend directory
cd frontend
npm run build
netlify deploy --prod --dir=dist
```

---

## Alternative: GitHub Pages with GitHub Actions

### Setup

1. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: **GitHub Actions**

2. **Update your Personal Access Token** to include `workflow` scope:
   - Go to: https://github.com/settings/tokens
   - Click your token → Edit
   - Check "workflow" scope
   - Update token in your local git credentials

3. **Push the GitHub Actions workflow**:
   ```bash
   git push
   ```

The `.github/workflows/deploy-frontend.yml` file will handle automatic deployment.

---

## Local Development

To test locally:

```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

To use local API instead of production:
1. Change `API_URL` in both predictor components to `'http://localhost:8080'`
2. Run `python api.py` from root directory

## Production URLs

- **Backend API**: https://wisteria-ctr-studio-azlh47c4pq-uc.a.run.app
- **Frontend**: Will be provided after Netlify deployment
