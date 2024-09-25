# PDF Embedding and Retrieval Example
This project demonstrates a working example of how to embed a large PDF file and retrieve it for user queries in an optimized way.

## Overview
The project consists of two main components:

1. `src/storeData.ts`: Processes a PDF file, extracts relevant information, and stores it in a vector database.
2. `src/queryData.ts`: Retrieves information from the vector database based on user queries and generates responses.

## Features
- Efficient embedding of large PDF files
- Custom text splitting for specific document structures
- Vector storage using HNSWLib
- Multi-query generation for comprehensive information retrieval
- Chunked processing of retrieved documents
- Answer aggregation for multiple responses

## Usage
1. Ensure you have the necessary dependencies installed.
2. Set up your OpenAI API key in the environment variables.
3. Run `npm run build` to build the project.
4. Run `npm run store` to process and store the PDF data.
5. Use `npm run query` to query the stored information.

## Note
This is a simplified example and may require additional setup and configuration for production use.