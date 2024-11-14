from inference_worker.core import abstraction
from inference_worker.core.exceptions import InferenceError
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter
from .model import Model
import logging
logger = logging.getLogger(__name__)


class ModelBuilder(abstraction.ModelBuilder):

    def build(self, model_file_paths, *args, **kwargs):
        model_file_path = model_file_paths[0]
        try:
            logger.info('[START] Model building started')
            model = YOLO(model_file_path)
            logger.info('Model Initialized')
            counter = ObjectCounter(model=model_file_path, region=[(40,500), (1200, 500)])
            logger.info('Object counter initialized')
            logger.info('[END] Model building finished')
            return Model(model, counter)
        except Exception as e:
            logger.exception('[FAILED] Failed building model')
            raise InferenceError('Model file cannot be loaded properly')

model_builder_class = ModelBuilder