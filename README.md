# Tuli AI - Advanced AI Assistant

An advanced AI system with enhanced knowledge processing, web scraping, and natural language understanding capabilities.

## Features

- Advanced natural language processing
- Dynamic knowledge base
- Web scraping capabilities
- Sentiment analysis
- Multi-source knowledge integration
- ChatGPT-like interface
- Code highlighting
- Markdown support
- Math equation support

## Quick Deployment

### Option 1: Render.com (Recommended)

1. Fork this repository
2. Go to [Render.com](https://render.com)
3. Sign up for a free account
4. Click "New +" and select "Web Service"
5. Connect your GitHub repository
6. Configure:
   - Name: tuli-ai
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn tuli_super:app`
7. Click "Create Web Service"

### Option 2: Railway.app

1. Fork this repository
2. Go to [Railway.app](https://railway.app)
3. Sign up with GitHub
4. Create new project
5. Select "Deploy from GitHub repo"
6. Configure:
   - Environment: Python
   - Start command: `gunicorn tuli_super:app`

### Option 3: Fly.io

1. Fork this repository
2. Go to [Fly.io](https://fly.io)
3. Sign up for a free account
4. Install flyctl:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
5. Login:
   ```bash
   flyctl auth login
   ```
6. Launch:
   ```bash
   flyctl launch
   ```
7. Deploy:
   ```bash
   flyctl deploy
   ```

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tuli-ai.git
   cd tuli-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python tuli_super.py
   ```

5. Access the application at `http://localhost:5000`

## API Endpoints

- `POST /chat`: Send messages to interact with Tuli AI
- `POST /learn`: Add new facts to the knowledge base
- `POST /scrape`: Scrape and process web content

## API Documentation

### /chat
- Method: POST
- Request body: `