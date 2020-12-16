import sys
import re
import json
import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.compat.v1.keras.models import load_model
import tensorflow.compat.v1.keras.backend as K
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.compat.v1.keras.applications import VGG16

model_fname = sys.argv[1]
print("Model path={}".format(model_fname))
save_pb_dir = sys.argv[2]
print("Save dir={}".format(save_pb_dir))
model_inputs_path = os.path.join(save_pb_dir, "model_inputs.json")
model_outputs_path = os.path.join(save_pb_dir, "model_outputs.json")
temp_create_model_path = os.path.join(save_pb_dir, "temp_create_model.h5")
temp_model_path = os.path.join(save_pb_dir, "temp_model.h5")
frozen_model_fname = "frozen_model.pb"

model = VGG16(weights=model_fname, include_top=True)
model.summary()
model.save(temp_create_model_path, save_format='h5')

# Load the model, but first re-save it with the learning phase flag shut off - this
# will get rid of the learning phase nodes prior to freezing the graph
session = tf.compat.v1.Session()
graph = session.graph
with session.as_default():
  with graph.as_default():
    model = load_model(temp_create_model_path)
    K.set_learning_phase(0)
    model.save(temp_model_path)

os.remove(temp_create_model_path)

# New session and graph for freezing
session = tf.compat.v1.Session()
graph = session.graph
with session.as_default():
  with graph.as_default():
    model = load_model(temp_model_path)
    K.set_learning_phase(0)
    INPUT_NODE = [t.op.name for t in model.inputs]
    with open(model_inputs_path, "w") as json_file:
      json.dump(INPUT_NODE, json_file)
    print("Wrote model inputs to {}".format(model_inputs_path))
    OUTPUT_NODE = [t.op.name for t in model.outputs]
    with open(model_outputs_path, "w") as json_file:
      json.dump(OUTPUT_NODE, json_file)
    print("Wrote model inputs to {}".format(model_outputs_path))
    print(INPUT_NODE, OUTPUT_NODE)
    graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
    graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, OUTPUT_NODE)
    graphdef_frozen = optimize_for_inference_lib.optimize_for_inference(
        graphdef_frozen, INPUT_NODE, OUTPUT_NODE, dtypes.float32.as_datatype_enum, toco_compatible=False)
    graph_io.write_graph(graphdef_frozen, save_pb_dir, 'frozen_model.pb', as_text=False)
    print("Wrote frozen graph to {}".format(os.path.join(save_pb_dir, frozen_model_fname)))

os.remove(temp_model_path)
