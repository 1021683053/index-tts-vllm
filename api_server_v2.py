import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import tempfile
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


async def save_upload_file(upload: UploadFile, output_path: str):
    with open(output_path, "wb") as output_file:
        while chunk := await upload.read(1024 * 1024):
            output_file.write(chunk)


def upload_suffix(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        return suffix
    return ".wav"


@app.post("/v1/audio/speech", responses={
    200: {"content": {"audio/wav": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}},
})
async def tts_api_form(
    text: str = Form(...),
    spk_audio: UploadFile = File(...),
    emo_control_method: int = Form(0),
    emo_audio: Optional[UploadFile] = File(None),
    emo_weight: float = Form(1.0),
    emo_vec: str = Form("[0, 0, 0, 0, 0, 0, 0, 0]"),
    emo_text: Optional[str] = Form(None),
    emo_random: bool = Form(False),
    max_text_tokens_per_sentence: int = Form(120),
):
    if emo_control_method not in {0, 1, 2, 3}:
        return JSONResponse(status_code=400, content={"error": "emo_control_method must be 0, 1, 2, or 3"})
    if emo_control_method == 1 and emo_audio is None:
        return JSONResponse(status_code=400, content={"error": "emo_audio is required when emo_control_method=1"})
    if emo_control_method == 3 and not emo_text:
        return JSONResponse(status_code=400, content={"error": "emo_text is required when emo_control_method=3"})

    try:
        vector = None
        if emo_control_method == 2:
            vector = json.loads(emo_vec)
            if not isinstance(vector, list) or len(vector) != 8 or not all(isinstance(value, (int, float)) for value in vector):
                return JSONResponse(status_code=400, content={"error": "emo_vec must be a JSON array containing 8 numbers"})
            if sum(vector) > 1.5:
                return JSONResponse(status_code=400, content={"error": "The sum of emo_vec must not exceed 1.5"})

        with tempfile.TemporaryDirectory(prefix="indextts2-upload-") as temp_dir:
            speaker_path = os.path.join(temp_dir, "speaker" + upload_suffix(spk_audio))
            await save_upload_file(spk_audio, speaker_path)

            emotion_path = None
            if emo_audio is not None:
                emotion_path = os.path.join(temp_dir, "emotion" + upload_suffix(emo_audio))
                await save_upload_file(emo_audio, emotion_path)

            sr, wav = await tts.infer(
                spk_audio_prompt=speaker_path,
                text=text,
                output_path=None,
                emo_audio_prompt=emotion_path if emo_control_method == 1 else None,
                emo_alpha=emo_weight if emo_control_method == 1 else 1.0,
                emo_vector=vector,
                use_emo_text=emo_control_method == 3,
                emo_text=emo_text,
                use_random=emo_random,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
            )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format="WAV")
            return Response(content=wav_buffer.getvalue(), media_type="audio/wav")
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "emo_vec must be valid JSON"})
    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})
    finally:
        await spk_audio.close()
        if emo_audio is not None:
            await emo_audio.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
