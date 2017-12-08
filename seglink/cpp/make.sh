TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared clip_rboxes_op.cc combine_segments_op.cc decode_segments_links_op.cc decode_local_rboxes_op.cc detection_mask_op.cc encode_groundtruth_op.cc polygons_to_rboxes_op.cc project_polygons_op.cc sample_crop_bbox_op.cc utilities.h -o libseglink.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

cp ./libseglink.so ../libseglink.so
