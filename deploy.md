# Render Deployment Guide

## Quick Deploy to Render

### 1. GitHub Setup
```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Document QA System"

# Add remote origin (replace with your actual GitHub repo URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2. Render Deployment
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect the configuration
5. Set environment variables in Render dashboard:
   - `API_BEARER_TOKEN`
   - `ASTRA_DB_APPLICATION_TOKEN`
   - `ASTRA_DB_ID`
   - `GROQ_API_KEY`

### 3. Environment Variables
Make sure to set these in Render:
- **API_BEARER_TOKEN**: Your API authentication token
- **ASTRA_DB_APPLICATION_TOKEN**: AstraDB application token
- **ASTRA_DB_ID**: Your AstraDB database ID
- **GROQ_API_KEY**: Your Groq API key

### 4. Deploy
Click "Create Web Service" and Render will automatically deploy your application.

### 5. Access Your API
Your API will be available at: `https://your-service-name.onrender.com`

- API Endpoint: `POST /api/v1/hackrx/run`
- Documentation: `GET /docs`
- Health Check: `GET /docs`
