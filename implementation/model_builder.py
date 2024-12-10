from inference_worker.core.domain.inference import ModelBuilder
from inference_worker.core.domain.inference import InferenceFailure
from ultralytics import YOLO
from .model import Model
import logging

logger = logging.getLogger(__name__)


class ModelBuilder(ModelBuilder):

    def build(self, model_file_paths, *args, **kwargs):
        modelFilePath = model_file_paths[0]
        try:
            logger.info('[START] Model building started')
            model = YOLO(modelFilePath)
            logger.info('Model Initialized')
            
            logger.info('[END] Model building finished')
            return Model(model, modelFilePath)
        except Exception as e:
            logger.exception('[FAILED] Failed building model')
            raise InferenceFailure(detail='Model file cannot be loaded properly')

model_builder_class = ModelBuilder