#!/bin/bash

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"

# Create GitHub repository (you'll need to do this manually)
echo "Please create a new repository on GitHub and then run:"
echo "git remote add origin YOUR_GITHUB_REPO_URL"
echo "git push -u origin main"

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

echo "Setup complete! Now you can deploy to your preferred platform:"
echo "1. Render.com (Recommended)"
echo "2. Railway.app"
echo "3. Fly.io"
echo "See README.md for detailed instructions." 