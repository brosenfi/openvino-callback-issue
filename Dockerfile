FROM openvino/ubuntu18_dev:latest

ENV INTEL_OPENVINO_DIR=/opt/intel/openvino

WORKDIR /home/openvino

ADD *.py *.jpg *.sh ./

# NOTE: Downgrade h5py which in this image still has this issue in this image = https://github.com/tensorflow/tensorflow/issues/44467 
RUN curl https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -o vgg16_weights_tf_dim_ordering_tf_kernels.h5 \
  && pip install 'h5py<3.0.0' \
  && python3 freeze_model.py ./vgg16_weights_tf_dim_ordering_tf_kernels.h5 ./ \
  && rm *.h5 \
  && ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_model.pb --reverse_input_channels --batch 1 \
  && rm frozen_model.pb

ENTRYPOINT ["/home/openvino/run.sh"]
