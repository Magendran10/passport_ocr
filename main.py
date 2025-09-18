#!/usr/bin/env python3
"""
Complete FastAPI Passport OCR Server with Google Gemini API Integration
Production-ready server with all functionality integrated.
"""

# New Dependency: pip install python-dotenv
import asyncio
import cv2
import numpy as np
import json
import logging
import time
import uuid
import io
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv # <-- NEW: Import the library
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from paddleocr import PaddleOCR
import google.generativeai as genai

# --- NEW: Load variables from the .env file ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration class now reads from the environment populated by .env
@dataclass
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    OCR_USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "15728640"))  # 15MB
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# Pydantic Models (unchanged)
class PersonalInfo(BaseModel):
    surname: Optional[str] = None
    given_names: Optional[str] = None
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    sex: Optional[str] = None
    place_of_birth: Optional[str] = None

class DocumentInfo(BaseModel):
    passport_number: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None
    issuing_country: Optional[str] = None
    issuing_authority: Optional[str] = None

class MRZData(BaseModel):
    document_type: Optional[str] = None
    issuing_country: Optional[str] = None
    passport_number: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    date_of_expiry: Optional[str] = None

class ProcessingResult(BaseModel):
    request_id: str
    filename: str
    personal_info: PersonalInfo
    document_info: DocumentInfo
    mrz_data: MRZData
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, Any]
    version: str = "2.0.0"

class BatchResult(BaseModel):
    batch_id: str
    total_files: int
    successful: int
    failed: int
    results: List[ProcessingResult]
    total_time: float

# Image Processor (unchanged)
class ImageProcessor:
    @staticmethod
    def process(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        metadata = {"steps_applied": [], "original_shape": image.shape}
        try:
            height, width = image.shape[:2]
            if width > 1200 or height > 800:
                scale = min(1200/width, 800/height)
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
                metadata["steps_applied"].append("resize")
            if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else: gray = image
            quality_score = ImageProcessor._assess_quality(gray)
            metadata["quality_score"] = quality_score
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            return sharpened, metadata
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
            metadata["error"] = str(e)
            return image, metadata

    @staticmethod
    def _assess_quality(gray: np.ndarray) -> float:
        try:
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()
            sharpness_score = min(sharpness / 100, 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(contrast / 50, 1.0)
            return (sharpness_score + brightness_score + contrast_score) / 3
        except Exception: return 0.5


# Gemini API Client (unchanged)
class GeminiClient:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.model_name = config.GEMINI_MODEL
        self.model = None

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in .env file. Gemini features will be disabled.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Successfully initialized Gemini client with model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.model = None

        self.system_prompt = """You are a specialized passport data extraction AI. Your task is to analyze OCR text from a passport and extract key information. Your response MUST be a single, valid JSON object and nothing else. Do not add any explanatory text before or after the JSON.
Rules for extraction:
1. Extract information only if it is clearly present in the text.
2. Use `null` for any fields that cannot be found.
3. Correct common OCR errors (e.g., mistaking '0' for 'O', '1' for 'I', '5' for 'S').
4. All dates must be standardized to the `DD/MM/YYYY` format.
5. The `sex` field must be either `M` or `F`.
6. provide all the names in the name column, don't spell correct the person names, give the names as it is.
The required JSON structure is:
{
  "personal_info": {"surname": "string or null", "given_names": "string or null", "date_of_birth": "DD/MM/YYYY or null", "nationality": "string or null", "sex": "M/F or null", "place_of_birth": "string or null"},
  "document_info": {"passport_number": "string or null", "date_of_issue": "DD/MM/YYYY or null", "date_of_expiry": "DD/MM/YYYY or null", "issuing_country": "string or null", "issuing_authority": "string or null"},
  "mrz_data": {"document_type": "string or null", "issuing_country": "string or null", "passport_number": "string or null", "nationality": "string or null", "date_of_birth": "DD/MM/YYYY or null", "sex": "M/F or null", "date_of_expiry": "DD/MM/YYYY or null"}
}"""

    async def parse_passport(self, ocr_text: str) -> Dict[str, Any]:
        if not self.model:
            logger.warning("Gemini client is not available. Using fallback regex parser.")
            return self._fallback_parse(ocr_text)
        try:
            prompt = f"{self.system_prompt}\n\nHere is the OCR text to process:\n---\n{ocr_text}\n---"
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
            response = await self.model.generate_content_async(prompt, generation_config=generation_config)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}. Falling back to regex parser.")
            return self._fallback_parse(ocr_text)

    def _fallback_parse(self, ocr_text: str) -> Dict[str, Any]:
        data = {"personal_info": {}, "document_info": {}, "mrz_data": {}}
        patterns = {
            "passport_number": r"(?:PASSPORT NO|DOCUMENTO NO|PASSEPORT NO)[:\s]*([A-Z0-9]{6,10})", "surname": r"(?:SURNAME|APELLIDO|NOM)[:\s]*([A-Z\s]+)", "given_names": r"(?:GIVEN NAMES|NOMBRES)[:\s]*([A-Z\s]+)",
            "nationality": r"(?:NATIONALITY|NACIONALIDAD)[:\s]*([A-Z]{3,20})", "sex": r"(?:SEX|SEXO)[:\s]*([MF])", "date_of_birth": r"(?:DATE OF BIRTH|BORN)[:\s]*([0-9/\-]{8,10})", "date_of_expiry": r"(?:DATE OF EXPIRY|EXPIRES)[:\s]*([0-9/\-]{8,10})"
        }
        text_upper = ocr_text.upper()
        for field, pattern in patterns.items():
            match = re.search(pattern, text_upper)
            if match:
                value = match.group(1).strip()
                if field in ["passport_number", "date_of_issue", "date_of_expiry", "issuing_country"]: data["document_info"][field] = value
                else: data["personal_info"][field] = value
        return data

    async def test_connection(self) -> bool:
        if not self.model: return False
        try:
            await self.model.generate_content_async("test", generation_config={"candidate_count": 1})
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

# Main Passport OCR Processor (unchanged)
class PassportOCRProcessor:
    def __init__(self):
        self.ocr = None
        self.image_processor = ImageProcessor()
        self.gemini_client = GeminiClient()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    async def initialize(self):
        try:
            loop = asyncio.get_event_loop()
            self.ocr = await loop.run_in_executor(self.thread_pool, lambda: PaddleOCR(use_angle_cls=True, lang='en', use_gpu=config.OCR_USE_GPU, show_log=False, use_mp=True, total_process_num=1))
            logger.info("OCR engine initialized successfully.")
            if await self.gemini_client.test_connection(): logger.info("Gemini API connection verified successfully.")
            else: logger.warning("Could not connect to Gemini API. Check your API key and network.")
        except Exception as e:
            logger.error(f"Fatal error during OCR engine initialization: {e}")
            raise

    async def process_image(self, image_bytes: bytes, filename: str) -> ProcessingResult:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        errors, warnings = [], []
        try:
            image = self._bytes_to_cv2_image(image_bytes)
            if image is None: raise ValueError("Cannot decode image file.")
            processed_image, img_metadata = self.image_processor.process(image)
            if img_metadata.get("quality_score", 0.0) < 0.3: warnings.append("Low image quality detected, may affect accuracy.")
            loop = asyncio.get_event_loop()
            ocr_results = await loop.run_in_executor(self.thread_pool, lambda: self.ocr.ocr(processed_image, cls=True))
            if not ocr_results or not ocr_results[0]: raise ValueError("No text could be detected in the image.")
            ocr_text = self._extract_text(ocr_results[0])
            parsed_data = await self.gemini_client.parse_passport(ocr_text)
            normalized_data = self._normalize_data(parsed_data)
            errors.extend(self._validate_data(normalized_data))
            confidence = self._calculate_confidence(ocr_results[0], normalized_data)
            processing_time = time.time() - start_time
            return ProcessingResult(
                request_id=request_id, filename=filename, personal_info=PersonalInfo(**normalized_data.get("personal_info", {})),
                document_info=DocumentInfo(**normalized_data.get("document_info", {})), mrz_data=MRZData(**normalized_data.get("mrz_data", {})),
                confidence=confidence, processing_time=processing_time, errors=errors, warnings=warnings
            )
        except Exception as e:
            processing_time = time.time() - start_time
            errors.append(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Processing failed for {filename}: {e}", exc_info=True)
            return ProcessingResult(
                request_id=request_id, filename=filename, personal_info=PersonalInfo(), document_info=DocumentInfo(), mrz_data=MRZData(),
                confidence=0.0, processing_time=processing_time, errors=errors, warnings=warnings
            )

    def _bytes_to_cv2_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None: return image
            pil_image = Image.open(io.BytesIO(image_bytes))
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to decode image bytes: {e}")
            return None

    def _extract_text(self, ocr_results: List) -> str:
        return "\n".join([line[1][0].strip() for line in ocr_results if len(line) >= 2 and line[1][1] > 0.4])

    def _normalize_data(self, data: Dict) -> Dict:
        normalized = {"personal_info": {}, "document_info": {}, "mrz_data": {}}
        for section, values in data.items():
            if section in normalized and isinstance(values, dict):
                for key, value in values.items():
                    if value and str(value).lower() != "null":
                        normalized[section][key] = self._normalize_field(key, str(value))
        return normalized

    def _normalize_field(self, field_name: str, value: str) -> Optional[str]:
        value = value.strip()
        if not value: return None
        if "date" in field_name: return self._normalize_date(value)
        if field_name == "passport_number": return re.sub(r'[^A-Z0-9]', '', value.upper())
        if field_name in ["surname", "given_names"]: return re.sub(r'[^A-Za-z\s-]', '', value).title().strip()
        if field_name == "sex": return value.upper()[0] if value.upper() and value.upper()[0] in ['M', 'F'] else None
        return value

    def _normalize_date(self, date_str: str) -> Optional[str]:
        cleaned = re.sub(r'[^\d/\-\s]', '', date_str)
        patterns = [r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', r'(\d{2})(\d{2})(\d{4})']
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                groups = match.groups()
                if len(groups[0]) == 4: year, month, day = groups
                else: day, month, year = groups
                if len(year) == 2: year = f"20{year}" if int(year) <= 30 else f"19{year}"
                return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        return None

    def _calculate_confidence(self, ocr_results: List, parsed_data: Dict) -> float:
        try:
            ocr_confidences = [line[1][1] for line in ocr_results if len(line) >= 2]
            avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
            all_fields, filled_fields = 0, 0
            for section in ["personal_info", "document_info"]:
                for value in parsed_data.get(section, {}).values():
                    all_fields += 1
                    if value: filled_fields += 1
            completeness_score = filled_fields / all_fields if all_fields > 0 else 0.0
            return (avg_ocr_confidence * 0.6 + completeness_score * 0.4)
        except Exception: return 0.5

    def _validate_data(self, data: Dict) -> List[str]:
        errors = []
        if not data.get("personal_info", {}).get("surname"): errors.append("Surname could not be determined.")
        if not data.get("document_info", {}).get("passport_number"): errors.append("Passport number could not be determined.")
        sex = data.get("personal_info", {}).get("sex")
        if sex and sex not in ["M", "F"]: errors.append(f"Invalid value for sex: '{sex}'")
        return errors

# Global state
processor: Optional[PassportOCRProcessor] = None

# FastAPI Application setup (unchanged)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    logger.info("Application startup...")
    processor = PassportOCRProcessor()
    await processor.initialize()
    yield
    logger.info("Application shutdown...")
    if processor and processor.thread_pool:
        processor.thread_pool.shutdown(wait=True)

app = FastAPI(
    title="Passport OCR API with Google Gemini",
    description="An API for extracting passport data using PaddleOCR and Gemini for intelligent parsing.",
    version="2.0.0",
    lifespan=lifespan
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")

# API Routes (unchanged)
@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    ocr_ok = processor is not None and processor.ocr is not None
    gemini_ok = await processor.gemini_client.test_connection()
    status = "healthy" if ocr_ok and gemini_ok else "degraded"
    return HealthResponse(
        status=status, timestamp=time.time(),
        services={"ocr_engine_initialized": ocr_ok, "gemini_api_connection": gemini_ok, "gemini_model": config.GEMINI_MODEL}
    )

@app.post("/process", response_model=ProcessingResult, summary="Process a Single Passport Image")
async def process_passport(file: UploadFile = File(...)):
    validate_file_upload(file)
    content = await file.read()
    if len(content) > config.MAX_FILE_SIZE: raise HTTPException(status_code=413, detail="File size exceeds limit.")
    return await processor.process_image(content, file.filename)

@app.post("/batch", response_model=BatchResult, summary="Process Multiple Passport Images")
async def batch_process(files: List[UploadFile] = File(...)):
    if len(files) > 10: raise HTTPException(status_code=400, detail="Max 10 files per batch.")
    start_time = time.time()
    tasks = []
    for file in files:
        validate_file_upload(file)
        content = await file.read()
        if len(content) > config.MAX_FILE_SIZE:
            failed_result = ProcessingResult(
                request_id=str(uuid.uuid4()), filename=file.filename, personal_info=PersonalInfo(), document_info=DocumentInfo(),
                mrz_data=MRZData(), confidence=0.0, processing_time=0.0, errors=[f"File '{file.filename}' exceeds size limit."]
            )
            tasks.append(asyncio.create_task(asyncio.sleep(0, result=failed_result)))
        else: tasks.append(processor.process_image(content, file.filename))
    results = await asyncio.gather(*tasks)
    successful = sum(1 for r in results if not r.errors)
    return BatchResult(
        batch_id=str(uuid.uuid4()), total_files=len(files), successful=successful,
        failed=len(files) - successful, results=results, total_time=time.time() - start_time
    )

def validate_file_upload(file: UploadFile):
    if not file.filename: raise HTTPException(status_code=400, detail="No filename provided.")
    if Path(file.filename).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
        raise HTTPException(status_code=415, detail=f"Unsupported file type.")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})

# Main entry point (unchanged)
if __name__ == "__main__":
    import uvicorn
    if not config.GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY is not set in the .env file. The API will run but Gemini features will be disabled.\n")
    print("🚀 Starting Passport OCR API Server with Google Gemini...")
    print(f"🔗 API Docs available at: http://localhost:8000/docs")
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True, workers=1, log_level="info")