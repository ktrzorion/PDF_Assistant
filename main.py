from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil
import asyncio
from typing import List
from chatbot_internal import PersistentProjectProcessor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Replace with your folder paths and processor instance
PDF_FOLDER = "pdfs"
CACHE_FOLDER = "project_cache"
processor = PersistentProjectProcessor(PDF_FOLDER, cache_dir=CACHE_FOLDER)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.on_event("startup")
async def initialize_processor():
    """Initialize the PersistentProjectProcessor during API startup."""
    print("Initializing the project processor...")
    await processor.initialize()
    print("Initialization complete.")

@app.get("/query-projects/")
async def query_projects(query: str):
    """
    Endpoint to query projects.
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query string cannot be empty.")

        results = await processor.query_projects(query)
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Endpoint to upload a PDF file for processing.
#     """
#     try:
#         if not file.filename.endswith(".pdf"):
#             raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#         file_path = os.path.join(PDF_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         # Reinitialize processor after upload
#         await processor.initialize()
#         return {"detail": f"Uploaded and processed {file.filename}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/projects/")
# async def list_projects():
#     """
#     Endpoint to list all processed projects.
#     """
#     try:
#         projects = [
#             {
#                 "name": project.name,
#                 "technologies": list(project.technologies),
#                 "description": project.description,
#                 "developers": list(project.developers),
#                 "platforms": list(project.platforms),
#                 "file_source": project.file_source,
#                 "pages": list(project.page_numbers),
#             }
#             for project in processor.projects.values()
#         ]
#         return JSONResponse(content={"projects": projects})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/process-pdfs/")
# async def process_pdfs():
#     """
#     Endpoint to process all PDFs in the specified folder.
#     """
#     try:
#         await processor.process_pdfs()
#         await processor.save_cached_data()
#         return {"detail": "Processed all PDFs and updated cache."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
