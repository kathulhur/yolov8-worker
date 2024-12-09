from inference_worker.core.domain import abstraction
from inference_worker.core.storage import default_storage
from inference_worker.core.domain.errors import InferenceError
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.solutions import ObjectCounter
from collections import defaultdict
import logging, mimetypes, pathlib, subprocess, cv2

logger = logging.getLogger(__name__)
class Model(abstraction.Model):

    def __init__(self, model: YOLO, modelFilePath):
        self.model = model
        self.modelFilePath = modelFilePath

    def infer(self, input_file_paths, *args, objectDetection=None, vehicleCounting=None, **kwargs):
        
        if objectDetection:
            confidence = objectDetection.get('confidence', 0.3)
            iou = objectDetection.get('iou', 0.7)
            imageSize = objectDetection.get('imageSize', 640)
            
            return self.handle_image_inference(input_file_paths, confidence=confidence, iou=iou, imageSize=imageSize)

        elif vehicleCounting:
            confidence = vehicleCounting.get('confidence', 0.3)
            iou = vehicleCounting.get('iou', 0.7)
            imageSize = vehicleCounting.get('imageSize', 640)
            lineCoordinates = vehicleCounting.get('lineCoordinates', '40,500,1200,500')

            return self.handle_video_inference(input_file_paths, confidence=confidence, iou=iou, imageSize=imageSize, lineCoordinates=lineCoordinates)

    def handle_image_inference(self, input_file_paths, *args, confidence=0.3, iou=0.7, imageSize=640, **kwargs):
        inputFilePath = input_file_paths[0]
        inputFileExtension = pathlib.Path(inputFilePath).suffix
        try:
            results: Results = self.model(inputFilePath, conf=confidence, iou=iou, imgsz=imageSize, show=False)
            objects_count = defaultdict(int)
            
            outputFile = default_storage.get_temporary_file(extension=inputFileExtension)
            for result in results:
                result.save(filename=outputFile.name)
                for box in result.boxes:
                    classId = int(box.cls)
                    className = self.model.names[classId]
                    objects_count[className] += 1

            return {
                'output': outputFile.name, 
                'metadata': {
                    'type': 'image/jpeg',
                    'objects': objects_count
                }
            }
        except Exception as e:
            print(e)
            raise InferenceError('Cannot make prediction. Please check your inputs.')

    def handle_video_inference(self, input_file_paths, *args, confidence=0.3, iou=0.7, imageSize=640, lineCoordinates, **kwargs):
        
        x1, y1, x2, y2 = list(map(lambda x : int(float(x.strip())), lineCoordinates.split(',')))

        counter = ObjectCounter(
            model=self.modelFilePath,
            region=[(x1, y1), (x2, y2)]
        )
        inputFilePath = input_file_paths[0]

        try:
            logger.info('[START] Video inference process begins')
            capture = cv2.VideoCapture(inputFilePath)

            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info('Line counter set')

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temporaryOutputFile = default_storage.get_temporary_file(extension='.mp4')
            temporaryOutputFilePath = temporaryOutputFile.name

            frame_rate = capture.get(cv2.CAP_PROP_FPS)

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_size = (frame_width, frame_height)

            videoOutput = cv2.VideoWriter(temporaryOutputFilePath, fourcc, frame_rate, frame_size)
            count = -1

            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    capture.release()
                    break
                count += 1

                print('frame', count, 'out of ', total_frames)
                modified_input = counter.count(frame)
            
                videoOutput.write(modified_input)


            videoOutput.release()
            logger.info('[END] Video inference process ends')
            
            outputFileName = default_storage.get_temp_file_name_candidate(file_extension='.mp4')
            self.convert_to_browser_friendly_format(temporaryOutputFile.name, outputFileName)
            return {
                'output': outputFileName,
                'metadata': {
                    'type': 'video/mp4',   
                    'counts': {
                            **counter.classwise_counts
                        }
                    }
                }
        except Exception as e:
            print(e)
            raise InferenceError('An error failed in the middle of inference.')

        finally:
            capture.release()

    def validate_confidence(self, value):
        try:
            value = float(value)
        except ValueError as e:
            raise InferenceError('Confidence value must be a floating point value and not any other data type.')

        if not isinstance(value, float):
            raise InferenceError('Confidence value must be a floating point value and not any other data type.')
        
        if value < 0.1 or value > 1.0:
                raise InferenceError('Confidence value must be between 0.1 and 1.0')
        
        return value

    def validate_iou(self, value):
        try:
            value = float(value)
        except ValueError as e:
            raise InferenceError('IoU value must be a floating point value and not any other data type.')
        
        if not isinstance(value, float):
                raise InferenceError('IoU value must be a floating point value and not any other data type.')
        
        if value < 0.1 or value > 1.0:
            raise InferenceError('IoU value must be between 0.1 and 1.0')
        
        return value
    
    def validate_imageSize(self, value):
        try:
            value = int(value)
        except ValueError as e:
            raise InferenceError('ImageSize value must be an integer and not any other data type.')
        

        if not isinstance(value, int):
            raise InferenceError('ImageSize value must be an integer and not any other data type.')
        
        if value < 320 or value > 1280:
                raise InferenceError('Image size must be greater than or equal to 320 and less than or equal to 1280')
        
        return value
        
        



    def convert_to_browser_friendly_format(self, input_path, output_path):
        command = [
            'ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path
        ]
        subprocess.run(command, check=True)
