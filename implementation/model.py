from inference_worker.core.domain.inference import Model
from inference_worker.core.application.storages.media import MediaFileStorage
from inference_worker.core.domain.inference import InferenceFailure, InferenceOutput
from ultralytics import YOLO
from ultralytics.engine.results import Results
from collections import defaultdict
import logging, pathlib

mediaFileStorage = MediaFileStorage()

logger = logging.getLogger(__name__)
class Model(Model):

    def __init__(self, model: YOLO, modelFilePath):
        self.model = model
        self.modelFilePath = modelFilePath

    def infer(self, input_file_paths, *args, standard={}, vehicleCounting=None, **kwargs):
        
        confidence = standard.get('confidence', 0.3)
        iou = standard.get('iou', 0.7)
        imageSize = standard.get('imageSize', 640)
            
        return self.handle_image_inference(input_file_paths, confidence=confidence, iou=iou, imageSize=imageSize)


    def handle_image_inference(self, input_file_paths, *args, confidence=0.3, iou=0.7, imageSize=640, **kwargs):
        inputFilePath = input_file_paths[0]
        inputFileExtension = pathlib.Path(inputFilePath).suffix
        try:
            results: Results = self.model(inputFilePath, conf=confidence, iou=iou, imgsz=imageSize, show=False)
            objects_count = defaultdict(int)
            
            outputFile = mediaFileStorage.create_empty_file(file_extension=inputFileExtension)
            for result in results:
                result.save(filename=outputFile.name)
                for box in result.boxes:
                    classId = int(box.cls)
                    className = self.model.names[classId]
                    objects_count[className] += 1

            return InferenceOutput(
                outputFilePath=outputFile.name, 
                metadata = {
                    'type': 'image/jpeg',
                    'objects': objects_count
                }
            )
        except Exception as e:
            print(e)
            raise InferenceFailure(detail='Cannot make prediction. Please check your inputs.')
