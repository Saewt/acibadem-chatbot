#!/bin/bash
set -e

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to start..."
sleep 5

echo "Pulling ${LLM_MODEL:-qwen3:4b}..."
ollama pull "${LLM_MODEL:-qwen3:4b}"

echo "Pulling ${EMBEDDING_MODEL:-nomic-embed-text}..."
ollama pull "${EMBEDDING_MODEL:-nomic-embed-text}"

echo "Models ready."
wait $SERVER_PID
