from threading import Lock

import pyrootutils
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.server.api_utils import MsgPackRequest, parse_args
from tools.server.exception_handler import ExceptionHandler
from tools.server.model_manager import ModelManager
from tools.server.views import (
    ASRView,
    ChatView,
    HealthView,
    TTSView,
    VQGANDecodeView,
    VQGANEncodeView,
)


class API(ExceptionHandler):
    def __init__(self):
        self.args = parse_args()
        self.app = FastAPI(
            title="Fish Speech API",
            version="1.5.0",
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.include_router(HealthView.router, prefix="/v1/health")
        self.app.include_router(VQGANEncodeView.router, prefix="/v1/vqgan/encode")
        self.app.include_router(VQGANDecodeView.router, prefix="/v1/vqgan/decode")
        self.app.include_router(ASRView.router, prefix="/v1/asr")
        self.app.include_router(TTSView.router, prefix="/v1/tts")
        self.app.include_router(ChatView.router, prefix="/v1/chat")

        # Add the state variables
        self.app.state.lock = Lock()
        self.app.state.device = self.args.device
        self.app.state.max_text_length = self.args.max_text_length

        # Associate the app with the model manager
        self.app.on_event("startup")(self.initialize_app)

    async def initialize_app(self):
        # Make the ModelManager available to the views
        self.app.state.model_manager = ModelManager(
            mode=self.args.mode,
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            asr_enabled=self.args.load_asr_model,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Startup done, listening server at http://{self.args.listen}")


if __name__ == "__main__":

    api = API()
    host, port = api.args.listen.split(":")

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=api.args.workers,
        log_level="info",
    )
