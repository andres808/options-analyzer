# Options Analysis App - Static Website

This directory contains a static website version of the Options Analysis application that can be deployed to GitHub Pages.

## What is this?

The static website version uses [Stlite](https://github.com/whitphx/stlite), which is a port of Streamlit that runs entirely in the browser without requiring a server. This allows you to deploy the app to GitHub Pages or any other static site hosting service.

## How to Deploy to GitHub Pages

1. Create a new GitHub repository for this project.

2. Initialize a git repository in this directory:
   ```
   cd static_site
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. Add your GitHub repository as a remote and push:
   ```
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

4. Enable GitHub Pages in your repository settings:
   - Go to your repository on GitHub
   - Navigate to "Settings" > "Pages"
   - Under "Source", select "main" branch and the "/ (root)" folder
   - Click "Save"

5. After a few minutes, your site will be available at `https://yourusername.github.io/your-repo-name/`

## File Structure

- `index.html` - The main Stlite HTML file that loads the Streamlit app in the browser
- `main_stlite.py` - The main Python file for the Streamlit app (adapted for Stlite)
- `data_fetcher.py` - Module for fetching stock and options data
- `analysis.py` - Module for analyzing options data
- `strategist.py` - Module for generating strategy recommendations
- `utils.py` - Utility functions for plotting and data manipulation
- `model.pkl` - Machine learning model file (placeholder in static version)

## Limitations

The static website version has some limitations compared to the full server version:

1. No data caching - all data is fetched fresh every time the app is loaded or refreshed
2. Slower performance - all processing happens in the browser
3. Limited memory - large datasets may cause browser performance issues
4. No background jobs - features like scheduled cache refresh are not available

## Customizing

You can customize the app by modifying the Python files. After making changes, simply update your repository:

```
git add .
git commit -m "Update app"
git push
```

GitHub Pages will automatically update your site with the new changes.