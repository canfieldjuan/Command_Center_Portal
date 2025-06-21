# AI Portal - Learning Machine v26.2.0

Advanced AI orchestration system with persistent learning capabilities, featuring multi-agent coordination, memory-enhanced planning, and adaptive execution.

## 🚀 Features

### Core AI Capabilities
- **Multi-Model AI Access**: OpenRouter integration for 20+ AI models
- **Google Gemini Integration**: Direct access to Google's latest models
- **Intelligent Routing**: Automatic model selection based on task complexity
- **Persona System**: Customizable AI personalities with specialized behaviors

### Advanced Orchestration
- **Memory-Enhanced Master Planner**: Learns from past execution patterns
- **Persona Dispatcher**: Selects optimal AI specialist for each task
- **Critic Agent**: Validates results and triggers adaptive corrections
- **Reflexive Swarm Architecture**: Self-improving multi-agent coordination

### Persistent Learning Memory
- **Plan Learning**: Stores successful execution strategies
- **Failure Analysis**: Learns from mistakes to avoid future errors
- **Task Success Patterns**: Builds expertise through repetition
- **Insight Storage**: Accumulates knowledge across sessions

### Tool Integration
- **Web Search**: Real-time information gathering via Serper API
- **Website Browsing**: Content extraction with Playwright
- **File Operations**: Secure workspace management
- **Ad Copy Generation**: Marketing content creation
- **Function Routing**: Automatic tool selection and execution

### Project Management
- **Multi-Project Support**: Organize work across different initiatives
- **Chat History**: Complete conversation tracking and analytics
- **User Management**: Multi-user support with isolated workspaces
- **Performance Metrics**: Detailed execution analytics

## 🏗️ Architecture

```
ai_portal/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── config.yaml         # Configuration settings
├── .env                # Environment variables
├── models/             # Database models (SQLAlchemy)
├── schemas/            # API schemas (Pydantic)
├── services/           # External service integrations
├── core/               # Core utilities and configuration
├── api/                # FastAPI route handlers
└── workspace/          # File operations workspace
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PostgreSQL database (Supabase recommended)
- API keys for AI services

### Quick Start

1. **Clone and Setup**
```bash
git clone <repository-url>
cd ai_portal
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

3. **Install Playwright**
```bash
playwright install chromium
```

4. **Run the Application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Environment Variables

**Required:**
- `SUPABASE_PASSWORD`: Your Supabase PostgreSQL password
- `OPENROUTER_API_KEY`: OpenRouter API key for multi-model access

**Optional:**
- `SERPER_API_KEY`: For web search functionality
- `COPYSHARK_API_TOKEN`: For ad copy generation
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google service account JSON

## 📚 API Documentation

### Core Endpoints

- `POST /chat` - Main conversation endpoint with tool integration
- `POST /objectives/execute` - Multi-agent orchestration
- `GET /system/status` - System health and configuration

### Project Management
- `POST /projects` - Create new project
- `GET /projects` - List user projects
- `GET /projects/{id}/history` - Get conversation history

### Persona Management
- `POST /personas` - Create AI personality
- `GET /personas` - List available personas
- `PUT /personas/{id}` - Update persona configuration

### Memory System
- `GET /memory/stats` - Learning system statistics
- `POST /memory/insights` - Store learned insights
- `GET /memory/plans/similar` - Query similar past plans

### System Monitoring
- `GET /health` - Basic health check
- `GET /system/status` - Comprehensive system status

Interactive API documentation: `http://localhost:8000/docs`

## 🧠 Memory Learning System

The AI Portal includes a sophisticated learning system that improves performance over time:

### Plan Learning
- Stores successful execution strategies
- Learns optimal task decomposition patterns
- Remembers effective model and tool combinations

### Failure Recovery
- Analyzes task failures and their corrections
- Builds knowledge of common error patterns
- Automatically suggests fixes for similar future failures

### Adaptive Execution
- Adjusts strategies based on past performance
- Selects optimal AI personas for specific tasks
- Continuously improves success rates

### Memory Statistics
```bash
curl http://localhost:8000/memory/stats
```

## 🔧 Configuration

### Model Tiers
- **Economy**: Fast, cost-effective models (GPT-3.5, Claude Haiku)
- **Standard**: Balanced performance (GPT-4, Claude Sonnet)
- **Premium**: Maximum capability (GPT-4 Turbo, Claude Opus)

### Task Routing
The system automatically routes tasks to appropriate models:
- Simple Q&A → Economy tier
- Code generation → Standard tier
- Complex reasoning → Premium tier
- Image generation → Specialized models

### Security Features
- Input validation and sanitization
- File type restrictions
- URL filtering for web browsing
- Path traversal protection
- Content size limits

## 🚦 Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "Explain quantum computing",
    "user_id": "user123",
    "task_type": "simple_qa"
})
```

### Tool-Enabled Request
```python
response = requests.post("http://localhost:8000/chat", json={
    "message": "Search for the latest AI news and save a summary to file",
    "user_id": "user123",
    "task_type": "auto"
})
```

### Multi-Agent Orchestration
```python
response = requests.post("http://localhost:8000/objectives/execute", json={
    "objective": "Research competitors, analyze their strategies, and create a market positioning report",
    "user_id": "user123"
})
```

### Custom Persona
```python
# Create specialist persona
persona_response = requests.post("http://localhost:8000/personas", json={
    "name": "Python Expert",
    "system_prompt": "You are a senior Python developer with expertise in backend systems...",
    "model_preference": "anthropic/claude-3-sonnet",
    "user_id": "user123"
})

# Use in conversation
chat_response = requests.post("http://localhost:8000/chat", json={
    "message": "Help me optimize this database query",
    "persona_id": persona_response.json()["id"],
    "user_id": "user123"
})
```

## 🔍 Monitoring and Analytics

### System Status
```bash
curl http://localhost:8000/system/status
```

### Performance Metrics
- Response times and success rates
- Model usage statistics
- Tool execution analytics
- Memory learning progress

### Chat History Analysis
- Conversation patterns and trends
- User engagement metrics
- Popular personas and models
- Error analysis and resolution

## 🛡️ Production Deployment

### Security Checklist
- [ ] Set strong `SECRET_KEY` in environment
- [ ] Configure CORS for your domain
- [ ] Use HTTPS in production
- [ ] Set up rate limiting
- [ ] Monitor API usage and costs
- [ ] Regular security updates

### Performance Optimization
- Database connection pooling (configured)
- Async request handling
- Memory system caching
- Response compression
- Static file serving

### Monitoring
- Structured logging with JSON output
- Health check endpoints
- Performance metrics collection
- Error tracking and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Code formatting
black ai_portal/
isort ai_portal/

# Linting
flake8 ai_portal/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: Check the `/docs` endpoint when running
- Issues: GitHub Issues for bug reports and feature requests
- API Reference: Interactive docs at `/docs` and `/redoc`

## 🎯 Roadmap

- [ ] Enhanced memory learning algorithms
- [ ] Multi-modal input support (images, audio)
- [ ] Workflow automation and scheduling
- [ ] Team collaboration features
- [ ] Advanced analytics dashboard
- [ ] Custom tool development framework

---

**AI Portal v26.2.0** - The future of intelligent automation is here! 🚀