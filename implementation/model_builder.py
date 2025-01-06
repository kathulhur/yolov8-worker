from inference_worker.workflow import ModelBuilder, InferenceFailure
from ultralytics import YOLO
from .model import YoloV8Model
import logging

logger = logging.getLogger(__name__)


class YOLOv8ModelBuilder(ModelBuilder):

    def build(self, model_file_paths, *args, **kwargs):
        modelFilePath = model_file_paths[0]
        try:
            logger.info('[START] Model building started')
            model = YOLO(modelFilePath)
            logger.info('Model Initialized')
            
            logger.info('[END] Model building finished')
            return YoloV8Model(model, modelFilePath)
        except Exception as e:
            logger.exception('[FAILED] Failed building model')
            raise InferenceFailure('Model file cannot be loaded properly')

model_builder_class = YOLOv8ModelBuilder