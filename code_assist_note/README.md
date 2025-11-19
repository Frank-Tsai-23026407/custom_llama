# AI Code Assist Guidelines

## 0. Enviroment
activate llama-env by `conda activate llama-env`

## 1. When creating note
**IMPORTANT**: All notes in this directory should be **concise and token-efficient**.

- Use bullet points instead of paragraphs
- Avoid redundant explanations
- Keep total length under 20 lines when possible

AI assistants reading these notes should extract essential information quickly without processing unnecessary text.

## 2. When createing plan

- Generate AI plan file under code_assist_note/plan
- The plan should include purpose and steps
- The purpose should be short, like "purpose:..."
- The steps should be short, like "1. ...\n 2. ..." and each step should less than ten words