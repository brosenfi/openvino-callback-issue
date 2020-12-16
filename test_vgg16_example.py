import cv2
import sys
import os
import json
import numpy as np
import pickle
import io
import base64
import threading, queue
import logging
from datetime import date, datetime, timezone, timedelta
from tensorflow.keras.preprocessing import image as image_utils
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from openvino.inference_engine import IENetwork, IECore, Blob

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s]: %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.propagate = False
logger.info("VGG16 test example started")

this_path = os.path.dirname(os.path.realpath(__file__))
device = os.getenv("TEST_DEVICE", "CPU")
model_xml_fname = "frozen_model.xml"
model_bin_fname = "frozen_model.bin"
logger.debug("Model file = {}".format(model_bin_fname))

ie = IECore()
net = ie.read_network(model=model_xml_fname, weights=model_bin_fname)
batch_size = int(os.getenv("BATCH_SIZE", "-1"))
if batch_size == -1:
    batch_size = net.batch_size
else:
    net.batch_size = batch_size
input_blob = next(iter(net.input_info))
n, c, h, w = net.input_info[input_blob].input_data.shape
logger.debug("n = {:d}, c = {:d}, h = {:d}, w = {:d}".format(n,  c,  h,  w))
out_blob = next(iter(net.outputs))
logger.debug("Input blob = {}, Output blob = {}".format(input_blob, out_blob))
num_requests_str = os.getenv("TEST_NUM_REQUESTS", "1")
if device == "CPU":
    network_config = {
        "CPU_THREADS_NUM": num_requests_str,
        "CPU_BIND_THREAD": "NUMA",
        "CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_NUMA"
    }
    num_requests = int(num_requests_str)
    if num_requests > 2:
        num_requests = 2
else:
    network_config = {}
    num_requests = int(num_requests_str)
exec_net = ie.load_network(network=net, device_name=device, config=network_config, num_requests=num_requests)
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', data_format='channels_first')

# Test images
files = os.listdir(".")
images = []
for name in files:
    if name.endswith('.jpg'):
        images.append(name)
        logger.debug("Image " + name + " detected")

if len(images) == 0:
    logger.debug("No images found in test directory.")
    sys.exit(1)

image_batches = []
for image_name in images:
    with open(os.path.join(".", image_name), 'rb') as img_file:
        in_memory_file = io.BytesIO(img_file.read())

    data8uint = np.frombuffer(in_memory_file.getvalue(), np.uint8)  # Convert string to an unsigned int array
    image = cv2.cvtColor(cv2.imdecode(data8uint, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image.shape[:-1] != (h, w):
        image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    image_with_n = np.expand_dims(image, 0)
    it = datagen.flow(image_with_n, batch_size=n)
    n_aug_samples = n - 1
    image_batch = np.ndarray(shape=(n, c, h, w), dtype=np.float32)
    image_batch[0] = image
    for i in range(n_aug_samples):
        image_batch[i + 1] = next(it)
    image_batch = preprocess_input(image_batch, data_format="channels_first", mode="tf")
    image_batch = Blob(net.input_info[input_blob].tensor_desc, image_batch)
    image_batches.append({"name": image_name, "batch": image_batch})


num_loops = int(os.getenv("NUMBER_OF_TEST_LOOPS", "1"))
def work_generator():
    for i in range(num_loops):
        for image_batch in image_batches:
            yield image_batch

work_queue = queue.Queue(maxsize=num_requests)
wait_for_work_timeout_seconds = int(os.getenv("WAIT_FOR_WORK_TIMEOUT", "60"))
def get_more_work(request_id):
    try:
        logger.debug("Request index {} getting more work from queue.".format(request_id))
        return work_queue.get(block=True, timeout=wait_for_work_timeout_seconds)
    except queue.Empty:
        logger.debug("Request index {} waited too long for work - quitting.".format(request_id))
        return None

requests_outstanding = 0
shutdown_condition = threading.Condition()
def signal_request_stream_exit(request_id):
    logger.debug("Request index {} shutting down.".format(request_id))
    global requests_outstanding
    with shutdown_condition:
        requests_outstanding -= 1
        if requests_outstanding == 0:
            shutdown_condition.notify()

class InferStreamWrapper:
    def __init__(self, request, id):
        self.init = True
        self.request = request
        self.id = id
        self.request.set_completion_callback(self.callback, self.id)
        
    def callback(self, status_code, user_data):
        if self.init:
            self.init = False
            logger.debug("Request index {} initial request complete.".format(self.id))
        else:
            logger.debug("Request index {} request complete.".format(self.id))
            work_queue.task_done()
        if (user_data != self.id):
            logger.error("Request ID {} does not correspond to user data {}".format(self.id, user_data))
        elif status_code != 0:
            logger.error("Request {} failed with status code {}".format(self.id, status_code))
        else:
            yhats = self.request.output_blobs[out_blob]
            meaned = np.mean(yhats.buffer, axis=0)
            meaned = np.expand_dims(meaned, axis=0)
        
        try:
            next_image_batch = get_more_work(self.id)
        except queue.Empty:
            logger.debug("Request index {} waited too long for work - quitting.".format(self.id))
            signal_request_stream_exit(self.id)
        else:
            if next_image_batch is None:
                work_queue.task_done()
                signal_request_stream_exit(self.id)
            else:
                self.request.set_blob(blob_name=input_blob, blob=next_image_batch["batch"])
                self.request.async_infer()
    
    def start(self, input_data):
        self.request.set_blob(blob_name=input_blob, blob=input_data["batch"])
        self.request.async_infer()

logger.debug("Length of requests is {}".format(str(len(exec_net.requests))))
wrapped_infer_streams = [InferStreamWrapper(request, index) for index, request in enumerate(exec_net.requests)]

with shutdown_condition:
    # Initiate the inference chains
    for infer_stream in wrapped_infer_streams:
        infer_stream.start(image_batches[0])
        requests_outstanding += 1
        
# Run the full test set
start_time = datetime.utcnow()
for test_image_batch in work_generator():
    try:
        work_queue.put(test_image_batch, block=False)
        logger.debug("Work put on the queue.")
    except queue.Full:
        # Wait for everything to finish before proceeding with the test
        # (simulating waiting for multiple requests to combine results)
        logger.debug("Full queue - waiting for everything to be processed.")
        work_queue.join()
        work_queue.put(test_image_batch, block=False)
        logger.debug("Work put on the queue.")

logger.debug("All of the test items put in the queue, waiting for completion.")
work_queue.join()
logger.debug("Everything completed, shutting down infer streams.")

with shutdown_condition:
    for infer_stream in wrapped_infer_streams:
        work_queue.put(None)
    if requests_outstanding != 0:
        shutdown_condition.wait(timeout=wait_for_work_timeout_seconds)

logger.debug("Infer streams are shut down.")

total_images_processed = len(images) * batch_size * num_loops
logger.info('test_example, message="Test complete - {} concurrent requests, did {} images in {}"'.format(
    num_requests_str, str(total_images_processed), str(datetime.utcnow() - start_time)))
