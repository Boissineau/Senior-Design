python generate_database_features.py \
--edge-image-file '../../data/train_data_91k.mat' \
--model-name 'network.pth' \
--batch-size 64 \
--cuda-id 0 \
--save-file '../../data/my_features/database_features.mat'

# python network_test.py \
# --edge-image-file '../../data/features/testset_feature2.mat' \
# --model-name 'network.pth' \
# --batch-size 64 \
# --cuda-id 0 \
# --save-file 'feature_camera_91k_test.mat'

#python network_test.py --edge-image-file '../../data/train_data_10k.mat' --model-name 'network.pth' --batch-size 64 --cuda-id 0 --save-file 'feature_camera_10k.mat'