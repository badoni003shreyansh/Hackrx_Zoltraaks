# Document QA System

A FastAPI-based application that processes PDF documents and answers questions using AI-powered language models. The system integrates with AstraDB for vector storage and Groq for language model inference.

## Features

- PDF document processing and text extraction
- Vector-based document search using AstraDB
- AI-powered question answering using Groq LLM
- RESTful API with authentication
- Optimized for performance and scalability

## Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: AstraDB with Cassandra
- **Embeddings**: Google Generative AI
- **Language Model**: Groq (Llama 3.3 70B)
- **PDF Processing**: PyPDF2
- **Deployment**: Render

## Prerequisites

- Python 3.11+
- AstraDB account and credentials
- Groq API key
- API Bearer token for authentication

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
API_BEARER_TOKEN=your_api_bearer_token
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
ASTRA_DB_ID=your_astra_db_id
GROQ_API_KEY=your_groq_api_key
```

## Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Hackrx_Zoltraaks
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your actual credentials
   ```

5. **Run the application**
   ```bash
   python index.py
   ```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### POST `/api/v1/hackrx/run`
Process a PDF document and answer questions.

**Headers:**
```
Authorization: Bearer <your_api_bearer_token>
```

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "Who are the key stakeholders?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The main topic is...",
    "The key stakeholders are..."
  ]
}
```

## Deployment

### Option 1: Render (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration

3. **Set Environment Variables**
   - In Render dashboard, go to your service
   - Navigate to "Environment" tab
   - Add all required environment variables



## Performance Optimizations

- **Text Chunking**: Optimized chunk size (600 chars) with overlap (100 chars)
- **Batch Processing**: Efficient vector store operations
- **Streaming**: PDF downloads use streaming for large files
- **Caching**: Vector embeddings are stored in AstraDB for reuse

## Monitoring and Health Checks

- Health check endpoint: `/docs`
- Automatic cleanup on shutdown
- Error logging and exception handling
- Request timeout configurations

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Ensure all required environment variables are set
   - Check `.env` file exists and is properly formatted

2. **AstraDB Connection Issues**
   - Verify `ASTRA_DB_APPLICATION_TOKEN` and `ASTRA_DB_ID`
   - Check network connectivity to AstraDB

3. **Groq API Errors**
   - Verify `GROQ_API_KEY` is valid
   - Check Groq service status

4. **PDF Processing Errors**
   - Ensure PDF URL is accessible
   - Check PDF file size and format

### Logs

Check application logs for detailed error information:
- Local: Console output
- Render: Service logs in dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the troubleshooting section above
