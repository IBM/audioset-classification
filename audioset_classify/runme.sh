#!/bin/bash
# You need to modify the dataset path. 
#DATA_DIR="/vol/vssp/msos/audioset/packed_features"

# You can to modify to your own workspace. 
WORKSPACE=`pwd`
#WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_audioset_classification"

BACKEND="keras"     # 'pytorch' | 'keras'

MODEL_TYPE="decision_level_multi_attention"    # 'decision_level_max_pooling'
                                                # | 'decision_level_average_pooling'
                                                # | 'decision_level_single_attention'
                                                # | 'decision_level_multi_attention'

# Train
python3 $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$RESULT_DIR --model_type=$MODEL_TYPE train

# Calculate averaged statistics. 
#python3 $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$RESULT_DIR --model_type=$MODEL_TYPE get_avg_stats
